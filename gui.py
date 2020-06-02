# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Graphical interface to the glint mask tools.
from functools import partial
from tkinter import *
from tkinter import filedialog, ttk, messagebox

from core.common import get_img_paths, process_imgs
from core.glint_mask import make_and_save_single_mask


class DirectoryPicker(ttk.Frame):
    def __init__(self, master, label, variable, callback=None):
        super().__init__(master)
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="black", background="white")

        self.master = master
        self.variable = variable
        self.callback = callback

        self.grid_rowconfigure(0, weight=1)
        ttk.Label(master=self, text=label, width=15).grid(row=0, column=0, sticky=E)

        self.grid_columnconfigure(1, weight=2)
        ttk.Label(master=self, textvariable=self.variable, style='BW.TLabel').grid(row=0, column=1, sticky=EW, padx=5)

        self.btn = ttk.Button(master=self, text="...", width=3, command=self._pick)
        self.btn.grid(row=0, column=2, sticky=E)

    def _pick(self):
        dir_name = filedialog.askdirectory()
        self.variable.set(dir_name)

        if self.callback is not None:
            self.callback(dir_name)


class GlintMaskApp(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        for x in range(3):
            self.columnconfigure(x, weight=1)

        for x in range(4):
            self.rowconfigure(x, weight=1)

        self.red_edge = BooleanVar()
        self.progress_val = IntVar()
        self.imgs_in = StringVar()
        self.masks_out = StringVar()

        self.chk_red_edge = ttk.Checkbutton(master=self, text="Red edge sensor", variable=self.red_edge)
        self.chk_red_edge.grid(row=0, sticky=W)
        self.chk_red_edge.bind('<Button-1>', lambda e: self.reset())

        self.picker_imgs_in = DirectoryPicker(self, label="In imgs dir.", variable=self.imgs_in,
                                              callback=lambda _: self.reset())
        self.picker_imgs_in.grid(row=1, columnspan=3, sticky=E + W)
        self.imgs_in.set(".ignore/imgs")

        self.picker_masks_out = DirectoryPicker(self, label="Out mask dir.", variable=self.masks_out,
                                                callback=lambda _: self.reset())
        self.picker_masks_out.grid(row=2, columnspan=3, sticky=E + W)
        self.masks_out.set(".ignore/out")

        self.progress = ttk.Progressbar(master=self, orient=HORIZONTAL, mode='determinate', variable=self.progress_val)
        self.progress.grid(row=3, columnspan=3, sticky=E + W)

        self.btn_process = ttk.Button(master=self, text="Generate", command=self.process)
        self.btn_process.grid(row=4, column=2, sticky=SE)

    @staticmethod
    def _err_callback(img_path, err):
        msg = '%r generated an exception: %s' % (img_path, err)
        messagebox.showinfo(message=msg)

    def _inc_progress(self, _):
        self.progress_val.set(self.progress_val.get() + 1)
        self.update_idletasks()

        if self.progress_val.get() == self.progress['maximum']:
            messagebox.showinfo(message='Processing complete')

    def reset(self):
        self.progress_val.set(0)
        self.btn_process.state = NORMAL
        self.picker_imgs_in.btn = NORMAL
        self.picker_masks_out.btn = NORMAL
        self.update_idletasks()

    def process(self):
        self.btn_process.state = DISABLED
        self.picker_imgs_in.btn = DISABLED
        self.picker_masks_out.btn = DISABLED

        red_edge = self.red_edge.get()
        img_files = get_img_paths(self.imgs_in.get(), self.masks_out.get(), red_edge=red_edge)

        self.progress_val.set(0)
        self.progress['maximum'] = len(img_files)

        f = partial(make_and_save_single_mask, mask_out_path=self.masks_out.get(), red_edge=red_edge)
        process_imgs(f, img_files, callback=self._inc_progress, err_callback=self._err_callback)


if __name__ == '__main__':
    root = Tk()
    root.resizable(True, True)
    root.rowconfigure(0, weight=2)
    root.columnconfigure(0, weight=2)

    app = GlintMaskApp(root, padding="12 3 12 3")
    app.grid(sticky=N + W + E + S)

    root.title("Glint Mask Generator")
    root.bind("<Return>", app.process)
    root.mainloop()
