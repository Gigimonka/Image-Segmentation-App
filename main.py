import tkinter as tk
from ui.app_interface import ImageSegmentationApp

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
