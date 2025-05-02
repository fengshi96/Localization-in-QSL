"""
Merge all PDFs named Akw_avg_{number}.pdf in the 'disorder' folder into a single
PDF (merged.pdf), and create an animation (animation.mp4 and animation.gif) by
converting each PDF to an image frame, ordered by the embedded number.

Dependencies:
  pip install PyPDF2 pdf2image imageio pillow
  (On macOS/Linux: install poppler for pdf2image, e.g. `brew install poppler`)

Usage:
  python merge_and_animate.py
"""

import os
import re
import glob
from PyPDF2 import PdfMerger
from pdf2image import convert_from_path
import imageio
import numpy as np
import fitz      # PyMuPDF
from PIL import Image
import imageio

def collect_and_sort_pdfs(pdf_dir="disorder"):
    """
    Find all files matching Akw_avg_{float}.pdf in pdf_dir, extract the float,
    and return a sorted list of tuples (time_float, filepath).
    """
    pattern = re.compile(r"Akw_avg_([0-9]+\.[0-9]+)\.pdf$")
    paths = glob.glob(os.path.join(pdf_dir, "Akw_avg_*.pdf"))
    entries = []
    for p in paths:
        fname = os.path.basename(p)
        m = pattern.match(fname)
        if m:
            t = float(m.group(1))
            entries.append((t, p))
    entries.sort(key=lambda x: x[0])
    return entries


def merge_pdfs(sorted_pdfs, output="merged.pdf"):
    """
    Merge all PDF files in sorted_pdfs (list of (time, path)) into one PDF.
    """
    merger = PdfMerger()
    for _, path in sorted_pdfs:
        merger.append(path)
    with open(output, "wb") as f_out:
        merger.write(f_out)
    merger.close()
    print(f"[+] Merged {len(sorted_pdfs)} PDFs into '{output}'.")


# def create_animation(sorted_pdfs, gif_out="animation.gif", fps=2, zoom=2):
#     frames = []
#     for t, pdf_path in sorted_pdfs:
#         doc = fitz.open(pdf_path)
#         page = doc[0]
#         # render at  zoom× scale (you can adjust zoom up/down)
#         mat = fitz.Matrix(zoom, zoom)
#         pix = page.get_pixmap(matrix=mat)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         frames.append(img)
#         doc.close()

#     # write GIF
#     imageio.mimsave(gif_out, frames, fps=fps)
#     print(f"Saved GIF: {gif_out}")


def create_animation(sorted_pdfs, gif_out="animation.gif", fps=2, dpi=50):
    """
    Create a GIF animation from a sorted list of (time, pdf_path) tuples.

    Parameters
    ----------
    sorted_pdfs : list of (float, str)
        Tuples of (time_stamp, path_to_pdf)
    gif_out : str
        Output GIF filename.
    fps : int or float
        Frames per second in the GIF.
    dpi : int
        Resolution (dots per inch) to render each PDF page.

    The function renders each PDF's first page at the given DPI and
    stitches them into a GIF.
    """
    frames = []
    for t, pdf_path in sorted_pdfs:
        doc = fitz.open(pdf_path)
        page = doc[0]
        # Rasterize at specified DPI
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        frames.append(img)
        doc.close()

    # Save as GIF
    imageio.mimsave(gif_out, frames, fps=fps)
    print(f"[+] Saved GIF animation to '{gif_out}' (DPI={dpi}, FPS={fps}).")



def main():
    pdf_dir = "disorder"
    sorted_pdfs = collect_and_sort_pdfs(pdf_dir)
    if not sorted_pdfs:
        print(f"[!] No PDFs found in '{pdf_dir}'. Exiting.")
        return

    # merge_pdfs(sorted_pdfs, output="merged.pdf")

    # determine fps based on number of frames or time differences:
    # here we simply use 1 frame per second
    fps = 10  
    dpi=50
    create_animation(sorted_pdfs,
                     gif_out="animation.gif", fps=fps, dpi=dpi)


if __name__ == "__main__":
    main()