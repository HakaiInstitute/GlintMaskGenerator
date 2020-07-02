"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-30
Description: Graphical interface to the glint mask tools.
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from core.bin_maskers import MicasenseRedEdgeMasker, DJIMultispectralMasker, BlueBinMasker
from core.specular_maskers import RGBSpecularMasker


class DirectoryPicker(ttk.Frame):
    """tkinter directory picker widget."""

    def __init__(self, master, label, variable, callback=None, label_width=None):
        super().__init__(master)
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="black", background="white")

        self.master = master
        self.variable = variable
        self.callback = callback

        self.grid_rowconfigure(0, weight=1)
        ttk.Label(master=self, text=label, width=label_width).grid(row=0, column=0, sticky='e')

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
    """Main widget for the Glint Mask Generator GUI."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.LABEL_WIDTH = 16
        self.DEFAULT_WORKERS = 5
        self.IMG_TYPES = {
            # Map display names to class required to process the images
            'RGB / CIR (Tom\'s method)': BlueBinMasker,
            'Micasense RedEdge (Tom\'s method)': MicasenseRedEdgeMasker,
            'DJI Multispectral (Tom\'s method)': DJIMultispectralMasker,
            'RGB / CIR (specular method)': RGBSpecularMasker
        }

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure(2, weight=1)

        for x in range(7):
            self.rowconfigure(x, weight=1, pad=5)

        self.img_type = tk.StringVar()
        self.img_type.set(list(self.IMG_TYPES.keys())[0])
        self.progress_val = tk.IntVar()
        self.imgs_in = tk.StringVar()
        self.masks_out = tk.StringVar()
        self.max_workers = tk.IntVar()
        self.max_workers.set(self.DEFAULT_WORKERS)

        # IMG TYPE SELECTION
        frm_img_type = tk.Frame(master=self)
        frm_img_type.grid(row=0, column=0, columnspan=3, sticky='ew')
        frm_img_type.grid_columnconfigure(1, weight=2)
        ttk.Label(master=frm_img_type, text="Imagery type", width=self.LABEL_WIDTH).grid(row=0, column=0, sticky='e')
        ttk.Combobox(master=frm_img_type, textvariable=self.img_type, values=list(self.IMG_TYPES.keys())) \
            .grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, ipady=5, pady=2)

        # IMG IN DIR
        self.picker_imgs_in = DirectoryPicker(self, label="Input images dir", variable=self.imgs_in,
                                              label_width=self.LABEL_WIDTH, callback=lambda _: self.reset())
        self.picker_imgs_in.grid(row=1, columnspan=3, sticky='ew')

        # MASK OUT DIR
        self.picker_masks_out = DirectoryPicker(self, label="Output masks dir", variable=self.masks_out,
                                                label_width=self.LABEL_WIDTH, callback=lambda _: self.reset())
        self.picker_masks_out.grid(row=2, columnspan=3, sticky='ew')

        # MAX WORKERS
        frm_max_workers = ttk.Frame(master=self)
        frm_max_workers.grid(row=3, column=0, columnspan=3, sticky='ew')
        frm_max_workers.grid_columnconfigure(1, weight=2)
        ttk.Label(master=frm_max_workers, text="Max workers", width=self.LABEL_WIDTH) \
            .grid(row=0, column=0, sticky='e', padx=2)
        self.spin_max_workers = ttk.Spinbox(master=frm_max_workers, from_=1, to=self.DEFAULT_WORKERS, increment=5,
                                            textvariable=self.max_workers)
        self.spin_max_workers.grid(row=0, column=1, columnspan=2, sticky='ew')

        # SEPARATOR
        ttk.Separator(master=self, orient="horizontal").grid(row=4, columnspan=3, sticky="ew", ipady=5)

        # PROGRESS
        self.progress = ttk.Progressbar(master=self, orient=tk.HORIZONTAL, mode='determinate',
                                        variable=self.progress_val)
        self.progress.grid(row=5, columnspan=3, sticky='ew', ipady=2, pady=2)

        # PROCESS BUTTON
        self.btn_process = ttk.Button(master=self, text="Generate", command=self.process)
        self.btn_process.grid(row=6, column=2, sticky='se', ipady=5)

    @staticmethod
    def _err_callback(img_path, err):
        msg = '%r generated an exception: %s' % (img_path, err)
        messagebox.showerror(message=msg)

    def _inc_progress(self, _):
        self.progress_val.set(self.progress_val.get() + 1)
        self.update_idletasks()

        if self.progress_val.get() == self.progress['maximum']:
            messagebox.showinfo(message='Processing complete')

    def reset(self):
        """Reset the state of the GUI for processing a new directory of images."""
        self.progress_val.set(0)
        self.btn_process.state = tk.NORMAL
        self.picker_imgs_in.btn = tk.NORMAL
        self.picker_masks_out.btn = tk.NORMAL
        self.update_idletasks()

    def process(self):
        """Process the images using the specified options and increment the progress bar."""
        self.btn_process.state = tk.DISABLED
        self.picker_imgs_in.btn = tk.DISABLED
        self.picker_masks_out.btn = tk.DISABLED
        self.progress_val.set(0)

        masker = self.IMG_TYPES[self.img_type.get()](self.imgs_in.get(), self.masks_out.get())

        if len(masker) <= 1:
            messagebox.showwarning(message="No files found in the given input directory.")
            return

        self.progress['maximum'] = len(masker)

        masker(max_workers=max(self.max_workers.get(), 1), callback=self._inc_progress,
               err_callback=self._err_callback)


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(True, True)
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.wm_minsize(width=500, height=220)

    app = GlintMaskApp(root, padding="12 3 12 3")
    app.grid(sticky='nsew')

    root.title("Glint Mask Generator")
    root.bind("<Return>", app.process)
    root.mainloop()
