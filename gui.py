# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Graphical interface to the glint mask tools.

from tkinter import *
from tkinter import filedialog
from tkinter import ttk

from core.common import get_img_paths
from core.glint_mask import red_edge_make_and_save_mask, rgb_make_and_save_mask


class DirectoryPicker(ttk.Frame):
    def __init__(self, master, label="Directory:"):
        super().__init__(master)
        ttk.Label(master=self, text=label, width=15).grid(row=0, column=0, sticky=E)
        self.ent = ttk.Entry(master=self, width=30)
        self.ent.grid(row=0, column=1, sticky=E + W)
        ttk.Button(master=self, text="...", width=3, command=self._pick).grid(row=0, column=2, sticky=E)

    def _pick(self):
        dir_name = filedialog.askdirectory()
        self.ent.delete(0)
        self.ent.insert(0, dir_name)

    @property
    def val(self):
        return self.ent.get().strip()


class GlintMaskApp(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.red_edge = BooleanVar()
        self.chk_red_edge = ttk.Checkbutton(master=self, text="Red edge sensor", variable=self.red_edge)
        self.chk_red_edge.grid(row=0, sticky=W)

        self.picker_imgs_in = DirectoryPicker(self, label="In imgs dir.")
        self.picker_imgs_in.grid(row=1, columnspan=3, sticky=E + W)

        self.progress = ttk.Progressbar(master=self, orient=HORIZONTAL, mode='determinate')
        self.progress.grid(row=3, columnspan=3, sticky=E + W)

        self.picker_masks_out = DirectoryPicker(self, label="Out mask dir.")
        self.picker_masks_out.grid(row=2, columnspan=3, sticky=E + W)

        btn_process = ttk.Button(master=self, text="Generate", command=self.process)
        btn_process.grid(row=4, column=2, sticky=SE)

    def _inc_progress(self, *args, **kwargs):
        if self.progress['value'] < self.progress['maximum']:
            self.progress['value'] += 1

    def process(self):
        img_files = get_img_paths(self.picker_imgs_in.val, self.picker_imgs_in.val, red_edge=self.red_edge)
        self.progress['maximum'] = len(img_files)

        if self.red_edge:
            red_edge_make_and_save_mask(self.picker_imgs_in.val, self.picker_masks_out.val, callback=self._inc_progress)
        else:
            rgb_make_and_save_mask(self.picker_imgs_in.val, self.picker_masks_out.val, callback=self._inc_progress)


if __name__ == '__main__':
    root = Tk()
    app = GlintMaskApp(root, padding="3 3 12 12")
    app.grid(sticky=N + W + E + S)
    root.title("Glint Mask Generator")
    root.bind("<Return>", app.process)
    root.mainloop()

    # TODO: Reactive layout
