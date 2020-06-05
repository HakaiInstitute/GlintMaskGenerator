# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Graphical interface to the glint mask tools.
import os
import tkinter as tk
from functools import partial
from tkinter import filedialog, ttk, messagebox

from core.common import get_img_paths, process_imgs
from core.glint_mask import make_and_save_single_mask as process_rgb
from core.specular_mask import make_and_save_single_mask as process_specular


class DirectoryPicker(ttk.Frame):
    def __init__(self, master, label, variable, callback=None):
        super().__init__(master)
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="black", background="white")

        self.master = master
        self.variable = variable
        self.callback = callback

        self.grid_rowconfigure(0, weight=1)
        ttk.Label(master=self, text=label).grid(row=0, column=0, sticky='e')

        self.grid_columnconfigure(1, weight=2)
        ent = ttk.Label(master=self, textvariable=self.variable, style='BW.TLabel')
        ent.grid(row=0, column=1, sticky='ew', padx=5, ipady=5, pady=2)

        self.btn = ttk.Button(master=self, text="...", width=3, command=self._pick)
        self.btn.grid(row=0, column=2, sticky='e')

    def _pick(self):
        dir_name = filedialog.askdirectory()
        self.variable.set(dir_name)

        if self.callback is not None:
            self.callback(dir_name)


class GlintMaskApp(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.DEFAULT_WORKERS = os.cpu_count() * 5

        for x in range(3):
            self.columnconfigure(0, weight=1)

        for x in range(6):
            self.rowconfigure(x, weight=1, pad=5)

        self.red_edge = tk.BooleanVar()
        self.progress_val = tk.IntVar()
        self.imgs_in = tk.StringVar()
        self.masks_out = tk.StringVar()
        self.max_workers = tk.IntVar()
        self.max_workers.set(self.DEFAULT_WORKERS)

        self.picker_imgs_in = DirectoryPicker(self, label="In imgs dir.", variable=self.imgs_in,
                                              callback=lambda _: self.reset())
        self.picker_imgs_in.grid(row=0, columnspan=3, sticky='ew')

        self.picker_masks_out = DirectoryPicker(self, label="Out mask dir.", variable=self.masks_out,
                                                callback=lambda _: self.reset())
        self.picker_masks_out.grid(row=1, columnspan=3, sticky='ew')

        self.chk_red_edge = ttk.Checkbutton(master=self, text="Red edge sensor", variable=self.red_edge)
        self.chk_red_edge.grid(row=2, column=0, sticky='w')
        self.chk_red_edge.bind('<Button-1>', lambda e: self.reset())

        self.lbl_max_workers = ttk.Label(master=self, text="Max workers").grid(row=2, column=1, sticky='e')
        self.spin_max_workers = ttk.Spinbox(master=self, from_=1, to=self.DEFAULT_WORKERS,
                                            textvariable=self.max_workers)
        self.spin_max_workers.grid(row=2, column=2, sticky='w')

        ttk.Separator(master=self, orient="horizontal").grid(row=3, columnspan=3, sticky="ew", ipady=5)

        self.progress = ttk.Progressbar(master=self, orient=tk.HORIZONTAL, mode='determinate',
                                        variable=self.progress_val)
        self.progress.grid(row=4, columnspan=3, sticky='ew', ipady=2, pady=2)

        self.btn_process = ttk.Button(master=self, text="Generate", command=self.process)
        self.btn_process.grid(row=5, column=2, sticky='se', ipady=5)

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
        self.btn_process.state = tk.NORMAL
        self.picker_imgs_in.btn = tk.NORMAL
        self.picker_masks_out.btn = tk.NORMAL
        self.update_idletasks()

    def process(self):
        self.btn_process.state = tk.DISABLED
        self.picker_imgs_in.btn = tk.DISABLED
        self.picker_masks_out.btn = tk.DISABLED

        red_edge = self.red_edge.get()
        in_dir = self.imgs_in.get()
        out_dir = self.masks_out.get()
        img_files = get_img_paths(in_dir, out_dir, red_edge=red_edge)
        max_workers = max(self.max_workers.get(), 1)

        self.progress_val.set(0)
        self.progress['maximum'] = len(img_files)

        if red_edge:
            f = partial(process_rgb, mask_out_path=out_dir, red_edge=True)
        else:
            f = partial(process_specular, mask_out_path=out_dir)
        process_imgs(f, img_files, max_workers=max_workers,
                     callback=self._inc_progress, err_callback=self._err_callback)


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(True, True)
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.wm_minsize(width=500, height=120)

    app = GlintMaskApp(root, padding="12 3 12 3")
    app.grid(sticky='nsew')

    root.title("Glint Mask Generator")
    root.bind("<Return>", app.process)
    root.mainloop()
