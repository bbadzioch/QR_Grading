import os
import subprocess
import io
import pyzbar.pyzbar as pyz
import numpy as np
import pdf2image
#import img2pdf
import PyPDF2 as pdf
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import resolve1
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
import tempfile
import shutil







def extract_pages(inputpdf, fpage, lpage):
    '''
    Extracts specified range of pages from a PyPDF2 PdfFileReader object.

    :inputpdf:
        A PyPDF2 PdfFileReader object.
    :fpage:
        Page number of the first page to be extracted.
    :lpage:
        Page number of the last page to be extracted.

    Returns:
        PyPDF2 PdfFileWriter object containing extracted pages
    '''
    output = pdf.PdfFileWriter()
    for i in range(fpage-1,lpage-1):
        output.addPage(inputpdf.getPage(i))
    return output



def pdf2pages(fname, output_fname=None, output_directory = None):
    '''
    Splits a pdf file into files containing individual pages

    :fname:
        Name of the pdf file.
    :output_fname:
        If string output files will be named output_fname_n.pdf where n is the page number.
        This argument can be also a function with signature f(fname, n, page) which returns a string.
        The page argument will be passed the PyPDF2 PdfFileWriter with the n-th page of the pdf file.
        If output_fname is a function, the output files will be named by return values of this function.
        Defaults to the name of the processed file.
    :output_directory:
        directory where output files will be saved. If the specified directory is does not exist it will
        be created.
        Defaults to the current working directory

    Returns:
        The list of file names created.

    Note: Page splitting seems to interfere with checkboxes embedded in pages.
    After splitting they can't be read, but if checkboxes are reselected they
    work again. Splitting pages using pdftk does not create this problem:
    os.system('pdftk merged.pdf burst > test.txt')
    '''

    # if no output_directory set it to the current dirtectory
    if output_directory == None:
         output_directory = os.getcwd()
    # is specified directory does not exist create it
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    if output_fname == None:
        output_fname = os.path.basename(fname)[:-4]

    if type(output_fname) == str:
        def label(n, page):
            s = f"{output_fname}_{n}.pdf"
            return s
    else:
        def label(n, page):
            return output_fname(fname, n, page)

    source = pdf.PdfFileReader(open(fname, 'rb'))
    num_pages = source.numPages
    outfiles = []
    for n in range(num_pages):
        page = extract_pages(source, n+1, n+2)
        outfile_name = label(n, page)
        outfile_path = os.path.join(output_directory, outfile_name)
        with open(outfile_path , "wb") as f:
            page.write(f)
        outfiles.append(outfile_name)
    return outfiles



def merge_pdfs(files, output_fname="merged.pdf"):
    '''
    Merge pdf files into a single pdf file.

    :files:
        A list of pdf file names.
    :output_fname:
        File name of the merged pdf file.

    Returns: None

    Note: If a pdf file is split into pages using pdf2pages, and then some pages
    are merged, then checkboxes will be unreadable due to an issue with pdf2pages.
    However, it seems that reselecting a few of the checkboxes in the merged file
    makes all of them readable again.
    '''

    output = pdf.PdfFileMerger()

    for f in files:
            output.append(f)
    with open(output_fname , "wb") as outpdf:
                output.write(outpdf)
    output.close()


def pdf_page2image(pdf_page, dpi=200):
    '''
    Converts a single pdf page into an image.
    :pdf_page:
        A PdfFileWriter object.
    :dpi:
        Resolution of the image produced from the pdf.
    Returns:
        PIL PpmImageFile object
    '''

    pdf_bytes = io.BytesIO()
    pdf_page.write(pdf_bytes)
    pdf_bytes.seek(0)
    page_image = pdf2image.convert_from_bytes(pdf_bytes.read(), dpi = dpi)[0]
    pdf_bytes.close()

    return page_image



def enhanced_qr_decode(img, xmax=5, ymax=5):
    """
    Enhanced decoder of QR codes. Can help with reading QR codes in noisy images.
    If a QR code is not found in the original image the function performs a series
    of morphological opening and closured on the image with various parametries in
    an attempty to enhance the QR code.

    :img:
        Numpy array encoding an image of the cover page.
        Note: matrix entries must be integers in the range 0-255
    :xmax:
    :ymax:
        Maximal values of parameters for computing openings and closures on the image.

    Returns:
        A list of pyzbar object with decoded QR codes. The list is empty if no codes
        were found.
    """

    qr = pyz.decode(img)

    # if QR code is not found, modify the image and try again
    if len(qr) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
        for i, j in [(i, j) for i in range(1, xmax+1) for j in range(1, ymax+1)]:
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((i, j)))
            opened = cv2.bitwise_not(opened)
            qr = pyz.decode(opened)
            if len(qr) != 0:
                break
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((i, j)))
            closed = cv2.bitwise_not(closed)
            qr = pyz.decode(closed)
            if len(qr) != 0:
                break
    return qr




def qr_decode_pdf(fname, data_only=True, dpi=200):
    '''
    Reads data from QR codes embedded in a pdf file.

    :fname:
        The name of the pdf file.
    :data_only:
        If true only data read from QR codes is returned, otherwise returns
        the whole pyzbar.decoded objects.
    :dpi:
        Resolution of the images produced from the pdf to read QR codes.

    Returns:
        list indexes by pages of the pdf where each list entry
        is a list of data read from QR codes on that page
    '''
    qr_data = []
    img = []
    with open(fname, 'rb') as f:
        source = pdf.PdfFileReader(f)
        num_pages = source.numPages
        for n in range(num_pages):
            output = pdf.PdfFileWriter()
            output.addPage(source.getPage(n))
            # Note: pdf2image.convert_from_bytes can convert
            # a multipage pdf file into a list of images, but
            # to save memory the code below reads one page at a time
            # io.BytesIO() provides a file objects to write the page to
            page = io.BytesIO()
            output.write(page)
            page.seek(0)
            page_image = pdf2image.convert_from_bytes(page.read(), dpi = dpi)[0]
            qr_list = pyz.decode(page_image)
            # if QR code is not found, modify the image and try again
            if len(qr_list) == 0:
                p = cv2.cvtColor(np.array(page_image), cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(p, 230, 255, cv2.THRESH_BINARY_INV)[1]
                for i, j in [(i, j) for i in range(1,6) for j in range(1, 6)]:
                    eroded = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((i, j)))
                    eroded = cv2.bitwise_not(eroded)
                    qr_list = pyz.decode(eroded)
                    if len(qr_list) != 0:
                        break

            if data_only:
                p_qr = [q.data.decode('utf8') for q in qr_list]
                qr_data.append(p_qr)
            else:
                qr_data.append(qr_list)
            page.close()
        return qr_data


def format_table(page, latex_template=None, maxpoints=10, name="XXXX", problem="0"):
    '''
    Formats a LaTeX template to add a score table to a given
    pdf page

    :page:
        Name of the pdf file to add score table to.
    :latex_template:
        Name of the LaTeX file with the formatting template.
    :maxpoints:
        The maximal numberber of points in the score table (up to 30 will look fine)
    :name:
        Name or id identiftying the student.
    :problem:
        The number of the problem corresponding to the score table

    Returns:
        A string with LaTeX code.
    '''

    default_template = "/Users/bb/Documents/GitHub/QR_Grading/grade_table_template.tex"

    if latex_template == None:
        latex_template = default_template


    # read the LaTeX template
    with open(latex_template, 'r') as f:
        latex = f.read()


    # insert data into the template
    shift = str((maxpoints + 2)/2) #for score table formatting
    latex = (latex.replace("FILENAME", page)
             .replace("MAXPOINTS", str(maxpoints))
             .replace("SHIFT", shift)
             .replace("PROBLEMNUM", problem)
             .replace("NAME", name)
            )
    return latex


def format_qr(page, qr_string, label_string, latex_template=None):
    '''
    Formats a LaTeX template to add QR code to a given
    pdf page

    :page:
        Name of the pdf file to QR code to.
    :qr_string:
        String to be encoded in the QR code.
    :label_string:
        String (up to 3 lines) with text of the label to be places next
        to the QR code.
    :latex_template:
        Name of the LaTeX file with the formatting template.

    Returns:
        A string with LaTeX code.
    '''

    default_template = "/Users/bb/Documents/GitHub/QR_Grading/qr_template.tex"

    if latex_template == None:
        latex_template = default_template


    # read the LaTeX template
    with open(latex_template, 'r') as f:
        latex = f.read()


    # insert data into the template
    latex = (latex.replace("FILENAME", page)
             .replace("QR_STRING", qr_string)
             .replace("QR_LABEL", label_string.replace('\n', '\\\\'))
            )
    return latex


def compile_latex(source, output_file = None, output_directory = None):
    '''
    Compiles a given string with LaTeX code into pdf  and cleans up all
    auxiliary files created in the process. Requires pdflatex to work.

    :source:
        String with LaTeX code to be compiled.
    :output_file:
        Name of the pdf file to be produced.
    :output_directory:
        Name of the directory where the pdf file will be saved.
        If none given the current directory will be used.

    Returns:
        A tuple consisting of the pdflatex subprocess return code and
    its stdout stream
    '''


    if output_directory == None:
        output_directory = os.getcwd()

    # remove output file suffix if needed
    if output_file[-4:] == ".pdf":
        output_file = output_file[:-4]

    tex_file_path = os.path.join(output_directory, output_file + ".tex")
    with open(tex_file_path, "w") as f:
        f.write(source)

    #compile LaTeX
    latex_command = ["pdflatex", "-shell-escape", "-output-directory", output_directory, output_file + ".tex"]
    completed = subprocess.run(latex_command, capture_output = True)

    # clean up the auxiliary files created during LaTeX compilation
    for f in os.listdir(output_directory):
        fl = f.split('.')
        if fl[0] == output_file and fl[-1] in ['tex', 'aux', 'log', 'gz', 'out']:
            os.remove(os.path.join(output_directory, f))

    return  completed.returncode, completed.stdout



def read_scores(fname):
    '''
    Gets data from checkbox forms embedded in a pdf file.

    :fname:
        Name of the pdf file.

    Returns:
        A list of names pdf checkboxed that are checked.

    Note: reading pdf form data with pdftk:
    os.system('pdftk source.pdf dump_data_fields_utf8 > output_file.txt')
    '''

    with open(fname, 'rb') as fp:
        parser = PDFParser(fp)
        doc = PDFDocument(parser)
        try:
            fields = resolve1(doc.catalog['AcroForm'])['Fields']
        except KeyError:
            return None

        scores = []
        for i in fields:
            field = resolve1(i)
            name, value = str(field.get('T')).split("'")[1], str(field['V']).split("'")[1]
            if value=="Yes":
                scores.append(name)
    ""
    return scores




def student_scores(score_list, problem_labels=None):

    '''
    Takes a list of names of pdf checkboxes indicating student scores
    and returns a pandas data frame of student scores, with rows corresponding
    to students and columns corresponding to problems. Also checks if
    multiple scores were entered for the same student and problem

    This assumes that checkbox names are of the form name.problem_label.score where:
    name = identifies the student
    problem_label = identifies the problem
    score = is the problem score for the student

    :score_list:
         List of names of pdf checkboxes indicating student scores.
    :problem_labels:
         List of numbers (or names) of all exam/assignment problems.
         If not given dataframe columns will be labeled by problem labels
         discovered in checkbox names.

    Returns:
        A tuple consisting of:
         - a pandas dataframe with scores
         - dictionary whose keys are names of students with multiple problem
           scores entered and whose values are lists of problems with multiple scores.
    '''

    if problem_labels == None:
        problem_d = {}
    else:
        problem_d  = dict.fromkeys([str(p) for p in problem_labels])

    score_d = {}
    multiple_scores  = {}
    for record in score_list:
        name, problem, score = record.split('.')
        if name not in score_d:
            score_d[name] = problem_d.copy()
        # check for multiple scores for a given problem, if they exist record them
        if problem in score_d[name] and score_d[name][problem] != None:
            if name not in multiple_scores:
                multiple_scores[name] = [problem]
                score_d[name] = {problem : "MULTI"}
            else:
                multiple_scores[name].append(problem)
        else:
            score_d[name][problem] = int(score)

    scores_df = pd.DataFrame(score_d).T
    return scores_df, multiple_scores


def qr_exam(fname, output_fname, qr_string, label_string, output_directory=None, latex_template=None):
    '''
    Embed QR codes in exam pages

    :fname:
        Name of the pdf file with the exam.
    :output_fname:
        Name of the output pdf file.
    :qr_string:
        A function with one integer argument n. The string returned by this function
        will be encoded in the QR code on page number n.
    :label_string:
        A function with one integer argument n. The string returned by this function
        Will printed next to the QR code on page number n. The string should consist
        of at most 3 lines.
    :output_directory:
        Name of the directory where the output pdf file will be saved. If none given
        the current directory will be used.
    :latex_template:
        The template file used for placing QR codes on pages, if None the default
        template will be used.

    Returns:
        Name of the output file.

    '''

    if latex_template == None:
        latex_template = "/Users/bb/Desktop/grading/qr_template.tex"

    # if no output_directory set it to the current dirtectory
    if output_directory == None:
         output_directory = os.getcwd()

    # split exam into pages
    page_list = pdf2pages(fname, output_fname=f"qr_temp_{fname[:-4]}", output_directory = output_directory)
    for n, p in enumerate(page_list):
        latex = format_qr(os.path.join(output_directory, p), qr_string(n), label_string(n))
        r = compile_latex(latex, output_file = "tex_" + p[:-4], output_directory=output_directory)
        if r[0] != 0:
            print("Latex compilation failed.")
            return r[1]
    qr_list = [os.path.join(output_directory, "tex_" + p) for p in page_list]
    merge_pdfs(qr_list, output_fname = os.path.join(output_directory, output_fname))
    for p in page_list:
        os.remove(os.path.join(output_directory, p))
    for p in qr_list:
        os.remove(p)

    return output_fname


def qr_exams_from_list(fname, id_list, exam_num, course_id, course_sec=0, output_directory=None, latex_template=None):
    '''
    Produces exams with embedded QR codes for a given list of student ids.

    :fname:
        Name of pdf file containg the exam.
    :id_list:
        List with ids of students.
    :exam_num:
        Number of exam or some short label identifying it (e.g. "FINAL").
    :course_id:
        ID of the course (e.g. "MTH 141").
    :course section:
        Course section (e.g. "Y").
    :output_directory:
        Name of the directory where exams will be created. If the directory does
        not exist it will be created. If the directory is not specified the current
        directory will be used.
    :latex_template:
        Template LaTeX file used to place the QR codes. If none given the default template
        will be used.

    Returns:
        A list with file names of produced pdf files.
    '''



    if output_directory == None:
        output_directory = os.getcwd()

    os.makedirs(output_directory, exist_ok = True)


    qr_str = f"{course_id}_{course_sec}_EX_{exam_num}"
    qr_lab = f"{course_id} SEC. {course_sec}\nEXAM {exam_num}"

    def qr_string_name(name, n):
        return f"{name}_{qr_str}_{n}"

    def label_string_name(name, n):
        return f"{name}\n{qr_lab} P.{n}"

    qr_exam_list = []

    for name in id_list:

        def qr_string(n):
            return qr_string_name(name, n)
        def label_string(n):
            return label_string_name(name, n)

        ex = qr_exam(fname = fname,
                    output_fname = f"{name}_{fname}",
                    qr_string = qr_string,
                    label_string = label_string,
                    output_directory = output_directory,
                    latex_template = latex_template
                   )

        qr_exam_list.append(ex)

    return qr_exam_list



def read_bubbles(img, show_plot=False, dilx=(4,10), dily=(10, 4)):
    '''
    Reads person number from the bubble sheet on the exam cover page

    :img:
        Numpy array encoding an image of the cover page
    :show_plot:
        If True displays plots illustating the image analysis.
        Useful for troubleshooting.
    :dilx:
    :dily:
        Tuples of two integers. They are used to dilate the image, making edges thicker which
        can help to find the contour of the bubble area. Dilations given by dilx and dily are
        applied to the image consecutively.


    Returns:
        An integer with the person number.
    '''


    def sort_corners(a):
        '''
        Given a 4x2 numpy array with vertices of a rectangle
        rearrange it, so that vertices appear in a clockwise
        order starting with the upper left.
        '''
        b = a.copy()
        ordering = [-1, -1, -1, -1]
        sa = np.sum(a,axis = 1)
        ordering[0] = np.argmin(sa)
        ordering[2] = np.argmax(sa)
        b[ordering[0]] = -1
        b[ordering[2]] = -1
        ordering[1] = np.argmax(b[:, 0])
        ordering[3] = np.argmax(b[:, 1])
        return a[ordering]



    if img.shape[2] > 3:
        img = img[:, :, :-1]
    if np.max(img) < 1.5:
        img = img*255
    img = img.astype("uint8")


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make grayscale
    gray = cv2.medianBlur(gray,5)  # blur to remove noise
    gray = cv2.Canny(gray, 75, 150) # find edges

    # thicken edges
    gray = cv2.dilate(gray, np.ones(dilx))
    gray = cv2.dilate(gray, np.ones(dily))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((5, 5)))


    # find the contour with the largest area
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    frame = cnts[0][:, 0, :]
    peri = cv2.arcLength(frame, True)
    approx = cv2.approxPolyDP(frame, 0.02 * peri, True)

    # apply perspective transformation to rectify the image within the countour
    pts1 = sort_corners(np.array(approx[:, 0, :], dtype = "float32"))
    pts2 = np.float32([[0,0],[800,0],[800,900],[0,900]])
    transf = cv2.getPerspectiveTransform(pts1,pts2)
    straight = cv2.warpPerspective(img, transf, (800, 900))

    # convert the rectified image to binary
    dst = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY)
    dst= cv2.threshold(dst, 220, 255, cv2.THRESH_BINARY)[1]

    # arrays with limints for subdividing the straightened
    # image into rows and columns
    x = np.linspace(18, 780, 9).astype(int)
    y = np.linspace(165, 860, 11).astype(int)

    # for each column find the row number with the lowest average pixel value
    selected = []
    for i in range(len(x) - 1):
        g = [int(np.mean(dst[y[j]:y[j+1], x[i]:x[i+1]]))  for j in range(len(y)-1)]
        selected.append(g.index(min(g)))

    # plots, just to check how the image analysis went
    if show_plot:
        plt.figure(figsize = (15, 5))
        plt.subplot(131)
        plt.xticks([])
        plt.yticks([])
        im = cv2.bitwise_not(gray)
        plt.imshow(im, cmap="gray")
        plt.fill(approx[:, 0, 0], approx[:, 0,  1], edgecolor='r', lw=3, fill=False)
        plt.subplot(132)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(dst, cmap="gray")
        plt.subplot(133)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(straight, cmap="gray")
        for i in range(len(x)-1):
            j = selected[i]
            plt.fill([x[i], x[i+1], x[i+1], x[i], x[i]],
                     [y[j], y[j], y[j+1], y[j+1], y[j]],
                     'r' , alpha = 0.3
                    )
        plt.show()

    person_number = sum([d*10**i for i, d in enumerate(selected[::-1])])

    return person_number




def pdfpage2img(page):

    f = io.BytesIO()
    page.write(f)
    f.seek(0)
    img = np.array(pdf2image.convert_from_bytes(f.read())[0])
    f.close()
    return img


def table_fname(name):
    return "t_" + name

def cover_page(qr):
    s = "P00"
    return s == qr.split("-")[-1]

def get_exam_code(qr):
    return qr[:-4]

def get_page_num(qr):
    return int(qr[-2:])

def has_table(fname):
    return fname.startswith("t_")

def name_without_table(name):
    if has_table(name):
        return name[2:]
    else:
        return name

def get_exam_name(qr):
    name = name_without_table(qr)
    name_list = name.split("-")
    return "-".join(name_list[:4])

def get_fname(f):
    fname = os.path.basename(f)
    fname = os.path.splitext(fname)[0]
    return fname


def covers_file(f):
    return "problem_0" in f




def read_scans(scans, gradebook, pages_dir, show_pnums=False):

    pnum_column = "person_number"
    qr_code_column = "qr_code"

    gradebook_df = pd.read_csv(gradebook)

    with open(scans, 'rb') as f:
        scanned_pdf = pdf.PdfFileReader(f)
        num_pages = scanned_pdf.numPages

        for n in range(num_pages):

            page = pdf.PdfFileWriter()
            page.addPage(scanned_pdf.getPage(n))
            page_image = pdfpage2img(page)

            # get QR code from the page
            qr_list = enhanced_qr_decode(page_image)

            qr_found = (len(qr_list) != 0)

            if qr_found:
                qr = qr_list[0].data.decode('utf8')

            # check if cover page, if so get the person number
            pnum_found = True
            if cover_page(qr):
                pnum = read_bubbles(page_image, show_plot=show_pnums)
                pnum_found = (pnum in gradebook_df[pnum_column].values)
                if pnum_found:
                    print(f"person_number: {pnum}")

            # if QR data or person number cannot be read ask for user input
            if not (qr_found and pnum_found):
                plt.figure(figsize = (15,20))
                plt.imshow(page_image)
                plt.show()
                print("\n\n")
                print(f"File: {scans}")
                print(f"Page: {n+1}")
                if not qr_found:
                    qr = input("QR code not found. \n\nEnter the exam code: ")
                    print("\n")
                if not pnum_found:
                    print(f"Person number has been read as: {pnum}.\nThis person number is not recognized.\n")
                    pnum = int(input("Enter person number: "))
                    print("\n")

            # write data to dataframes
            if cover_page(qr):
                i = np.flatnonzero(gradebook_df[pnum_column].values == pnum)[0]
                # record the exam code of a student in the gradebook
                gradebook_df.loc[i, qr_code_column] = get_exam_code(qr)
            else:
                i = np.flatnonzero(gradebook_df[qr_code_column].values == get_exam_code(qr))[0]
                pnum = (gradebook_df[pnum_column].values)[i]


            page_file = os.path.join(pages_dir, qr + ".pdf")
            with open(page_file , "wb") as f:
                page.write(f)
                print(qr)


    # save gradebook
    gradebook_df.to_csv(gradebook, index=False)




def add_score_table(pages_dir, gradebook, latex_template=None, maxpoints=10):

    pnum_column = "person_number"
    qr_code_column = "qr_code"

    if type(maxpoints) != list:
        maxpoints = [maxpoints]

    gradebook_df = pd.read_csv(gradebook)

    files = glob.glob(os.path.join(pages_dir, "*.pdf"))
    files = [f for f in files if not has_table(get_fname(f))]

    for f in files:

        fname = os.path.basename(f)
        name = get_fname(f)
        output_file = table_fname(fname)


        if cover_page(name):
            shutil.copy(f, os.path.join(pages_dir, output_file))
            continue

        page_num = get_page_num(name)
        max_score = maxpoints[min(get_page_num(name)-1, len(maxpoints)-1)]

        tex = format_table(f,
                           latex_template=None,
                           maxpoints=max_score,
                           name=get_exam_code(name),
                           problem=str(page_num)
                          )

        output_file = table_fname(fname)
        c, _ = compile_latex(tex, output_file = output_file , output_directory = pages_dir)
        print(f"{output_file}   -->   {c}")



def assemble_by_problem(pages_dir, grading_dir):

    files = glob.glob(os.path.join(pages_dir, "*.pdf"))
    files = [f for f in files if has_table(get_fname(f))]

    files_dir = {}

    for f in files:

        fname = os.path.basename(f)
        name, extension = os.path.splitext(fname)
        files_dir[f] = get_page_num(name)

    problems = set(files_dir.values())


    for n in problems:
        f_n = [f for f in files_dir if files_dir[f] == n]
        f_n.sort()
        fnames_n = [name_without_table(os.path.basename(f)) for f in f_n]
        exam_name = get_exam_name(fnames_n[0])
        output = f"{exam_name}_problem_{n}"


        output_fname = os.path.join(grading_dir, output + ".pdf")
        merge_pdfs(f_n , output_fname=output_fname)

        output_json = os.path.join(pages_dir, output + ".json")
        with open(output_json, 'w') as jfile:
            json.dump(fnames_n, jfile)



def prepare_grading(main_dir, show_pnums=False,  maxpoints=10):

    scans_dir = os.path.join(main_dir, "scans")
    gradebook = os.path.join(main_dir, "gradebook.csv")
    pages_dir = os.path.join(main_dir, "pages")
    grading_dir = os.path.join(main_dir, "for_grading")
    dest_dir = os.path.join(main_dir, "graded")


    if not os.path.isdir(pages_dir):
        os.makedirs(pages_dir)
    if not os.path.isdir(grading_dir):
        os.makedirs(grading_dir)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    print("\nReading scanned files... \n")

    files = glob.glob(os.path.join(scans_dir, "*.pdf"))
    for f in files:
        read_scans(f, gradebook=gradebook, pages_dir=pages_dir, show_pnums=show_pnums)

    print("\n\nAdding score tables... \n")
    add_score_table(pages_dir=pages_dir, gradebook=gradebook, maxpoints=maxpoints)

    print("\n\nAssembling files for grading... \n")
    assemble_by_problem(pages_dir=pages_dir, grading_dir=grading_dir)

    print("\nGrading files ready.")






def prepare_scores(grading_dir):
    dfs = []
    duplicates = []

    files = glob.glob(os.path.join(grading_dir, "*.pdf"))
    files = [f for f in files if not covers_file(get_fname(f))]

    for f in files:
        print(f"{os.path.basename(f)}    ", end="")
        score_list = read_scores(f)
        if score_list == None:
            print("no scores, omitted")
            continue
        else:
            print("ok")
        score_list.sort()
        df, _ = student_scores(score_list)
        dfs.append(df)

    if len(dfs) == 0:
        return None

    df_combined = pd.concat(dfs, sort=True, axis=1)
    problem_cols = df_combined.columns.tolist()
    df_combined["total"] = df_combined[problem_cols].sum(axis=1, numeric_only=True)
    df_combined.reset_index(inplace=True)
    df_combined.rename(columns={"index": "qr_code"}, inplace=True)

    return df_combined





def get_scores(main_dir, new_gradebook=None, save=True):

    grading_dir = os.path.join(main_dir, "for_grading")
    gradebook = os.path.join(main_dir, "gradebook.csv")


    qr_code_column = "qr_code"

    if new_gradebook == None:
        new_gradebook = gradebook

    scores_df = prepare_scores(grading_dir)
    if type(scores_df) == type(None):
        return None

    gradebook_df = pd.read_csv(gradebook)

    problem_cols = scores_df.columns.tolist()
    problem_cols.remove(qr_code_column)

    try:
        gradebook_df.drop(columns = problem_cols, inplace=True)
    except KeyError:
        pass

    gradebook_df  =  gradebook_df.merge(scores_df,
                                               how="left",
                                               on=qr_code_column
                                              )

    if save:
        gradebook_df.to_csv(new_gradebook, index=False)

    return scores_df, gradebook_df






def assemble_by_student(main_dir):

    gradebook = os.path.join(main_dir, "gradebook.csv")
    pages_dir = os.path.join(main_dir, "pages")
    grading_dir = os.path.join(main_dir, "for_grading")
    dest_dir = os.path.join(main_dir, "graded")

    qr_code_column = "qr_code"
    pnum_column = "person_number"
    total_column = "total"

    gradebook_df =  pd.read_csv(gradebook)
    files = glob.glob(os.path.join(grading_dir, "*.pdf"))

    temp_dir = tempfile.mkdtemp()
    temp_dir = "sample_grading/temp"

    for f in files:

        fname = os.path.basename(f)

        jfile_name = os.path.join(pages_dir, os.path.splitext(fname)[0] + ".json")
        with open(jfile_name, 'r') as jfile:
            name_list = json.load(jfile)

            def set_page_names(fname, n, page):
                return name_list[n]

        pdf2pages(f, output_fname=set_page_names, output_directory = temp_dir)


    #covers = [f for f in glob.glob(os.path.join(pages_dir, "*.pdf")) if cover_page(get_fname(f))]
    #for f in covers:
    #    shutil.copy(f, temp_dir)


    files = glob.glob(os.path.join(temp_dir, "*.pdf"))
    codes = set( get_exam_code(get_fname(f)) for f in files)
    for exam_code in codes:
        exam_pages = [f for f in files if get_exam_code(get_fname(f)) == exam_code]
        exam_pages.sort()
        output_fname = os.path.join(dest_dir, exam_code + ".pdf")
        merge_pdfs(exam_pages, output_fname=output_fname)

    shutil.rmtree(temp_dir)
