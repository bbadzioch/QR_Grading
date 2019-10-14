import os
import subprocess
import re
import itertools
import glob
import PyPDF2 as pdf
import pdf2image
import io
import pyzbar.pyzbar as pyz
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import resolve1



import tempfile
import shutil


'''
QR codes formatting scheme:
MTH-COURSE_NUM-SEMESTER_CODE-EXAM_NAME-EXAM_COPY_NUMBER-EXAM_PAGE_NUMBER

For example:
MTH-309T-F19-EX1-001-P02

BEFORE THE EXAM:
Exam preparation: Use the make_exams function (requires an exam template file).

PREPARATION FOR GRADING:
Exam scans need to be placed in a directory named "scans"
* A gradebook file must be a csv file with at least two columns:
  - a column named "person_number" with student UB person numbers
  - a column named "qr_code" which will be populated by the program
    by QR codes of the exam copies.
* Once thi is done use the prepare_grading function.
* Scanned exams will be diassebled into pdf files with induvidual exam pages
  and  saved in the directory "pages" (which is automatically created).
  Each exam page is named according to the scheme QR.pdf where QR is the QR code
  of the page.

AFTER GRADING:


'''


class GradingBase():

    '''
    Base class establishing file and directory structure used in grading.
    '''

    def __init__(self, main_dir = None, gradebook = None, init_grading_data=False):
        
        if main_dir is None:
            self.main_dir = os.getcwd()
        else:
            self.main_dir = main_dir


        self.scans_dir = os.path.join(self.main_dir, "scans")
        self.pages_dir = os.path.join(self.main_dir, "pages")
        self.for_grading_dir = os.path.join(self.main_dir, "for_grading")
        self.graded_dir = os.path.join(self.main_dir, "graded")

        if gradebook is None:
            self.gradebook = os.path.join(self.main_dir, "gradebook.csv")
        else:
            self.gradebook = gradebook

        self.missing_data_pages = os.path.join(self.scans_dir, "missing_data.pdf")

        self.grading_data_jfile = os.path.join(self.main_dir, "grading_data.json")
        
        self.init_grading_data = {"maxpoints": {},
                                  "processed_scans": [], 
                                  "page_lists" : {},
                                  "missing_data" : []
                                  }
        if init_grading_data:
                with open(self.grading_data_jfile) as foo:
                    json.dump(self.init_grading_data, foo)
                if os.path.isfile(self.missing_data_pages):
                    os.remove(self.missing_data_pages)
        
        self.pnum_column = "person_number"
        self.qr_code_column = "qr_code"
        self.pnum_column = "person_number"
        self.total_column = "total"
        self.grade_column = "grade"
        

    def get_grading_data(self):
        if os.path.isfile(self.grading_data_jfile):
            with open(self.grading_data_jfile) as foo:
                return json.load(foo)
        else:
            return self.init_grading_data


    def set_grading_data(self, data):
        with open(self.grading_data_jfile, 'w') as foo:
            json.dump(data, foo)


    def split_for_grading_files(self, dest_dir):

        files = glob.glob(os.path.join(self.for_grading_dir, "*_problem_*.pdf"))
        page_lists = self.get_grading_data()["page_lists"]

        for f in files:
            f_list = page_lists[os.path.basename(f)]
            def set_page_names(fname, n, page):
                return f_list[n]

            pdf2pages(f, output_fname=set_page_names, output_directory = dest_dir)



def pdfpage2img(pdf_page, dpi=200):
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
    page_image = np.array(pdf2image.convert_from_bytes(pdf_bytes.read(), dpi = dpi)[0])
    pdf_bytes.close()

    return page_image


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

    # if no output_directory set it to the current directory
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
    '''

    output = pdf.PdfFileMerger()

    for f in files:
            output.append(f)
    with open(output_fname , "wb") as outpdf:
                output.write(outpdf)
    output.close()


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




def compile_latex(source, output_file, output_directory = None):
    '''
    Compiles a given string with LaTeX code into pdf  and cleans up all
    auxiliary files created in the process. Requires pdflatex to work.

    :source:
        String with LaTeX code to be compiled.
    :output_file:
        Name of the pdf file to be produced.
    :output_directory:
        Name of the directory where the pdf file will be saved.
        If none given the current directory will be used. If the given directory
        does not exist, it will be created.

    Returns:
        A tuple consisting of the pdflatex subprocess return code and
    its stdout stream
    '''


    if output_directory == None:
        output_directory = os.getcwd()
    # create the output directory if needed
    if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

    output_file = os.path.splitext(os.path.basename(output_file))[0]
    tex_file_path = os.path.join(output_directory, output_file + ".tex")
    with open(tex_file_path, "w") as f:
        f.write(source)

    #compile LaTeX
    latex_command = ["pdflatex", "-shell-escape", "-output-directory", output_directory, output_file + ".tex"]
    completed = subprocess.run(latex_command, capture_output = True)

    # clean up the auxiliary files created during LaTeX compilation
    for f in os.listdir(output_directory):
        fl = os.path.splitext(f)
        if fl[0] == output_file and fl[-1] in ['.tex', '.aux', '.log', '.gz', '.out']:
            os.remove(os.path.join(output_directory, f))

    return  completed.returncode, completed.stdout



def make_exams(template, N, output_file=None, output_directory = None):
    '''
    Produces pdf files with copies of an exam from a given LaTeX template file

    :tamplate:
        Name of the LaTeX template file with the exam.
    :N:
       Integer. The number of copies of the exam to be produced.
    :output_file:
        Name of the pdf files to be produced; these files will be named
        output_file_num.pdf where num is the number of the exam copy.
        If  output_file is None, the template file name is used.
    :output_directory:
        Name of the directory where the pdf files will be saved.
        If none given the current directory will be used. If the given directory
        does not exist, it will be created.

    Returns:
        A tuple consisting of the pdflatex subprocess return code and
    its stdout stream
    '''

    if output_directory == None:
        output_directory = os.getcwd()
    # create the output direxctory if needed
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # if no name of the output file, use the template file name
    if output_file == None:
        output_file = os.path.splitext(os.path.basename(template))[0]

    with open(template) as foo:
        template_tex = foo.read()

    for n in range(1, N+1):
        print(f"Compiling: {n:03}\r", end="")
        pnum = itertools.count()
        exam_copy = re.sub(r"PLACEHOLDER", lambda x: f'{n:03}-P{next(pnum):02}', template_tex)
        c = compile_latex(exam_copy, output_file = f"{output_file}_{n:03}", output_directory = output_directory)
        # if LateX compilation fails, return LaTeX compilation log.
        if c[0] != 0:
            return c

    print("Done!"+ " "*30)
    return None


class ExamCode():

    '''
    Class defining methods for manipulating exam file names. 
    '''

    def __init__(self, code):
        self.code = code
        self.head, self.tail = os.path.split(self.code)
        self.base, self.ext = os.path.splitext(self.tail)

    def has_table(self):
        '''
        Checks if a given file name corresponds to a pdf file of an exam page
        with a score table added.
        '''
        return self.base.startswith("t_")
        

    def get_exam_code(self):
        '''
        Strips page number from an exam page QR code, leaving out the part
        identifying the exam copy.
        '''
        code =  "-".join(self.base.split("-")[:-1])
        if self.has_table():
            code = code[2:]
        return code

    def get_page_num(self):
        '''
        Returns the part of the exam page QR code giving the exam page number.
        '''
        return int(self.base.split("-")[-1][1:])

    def is_cover(self):
        '''
        Checks if a page is a cover page of an exam. 
        '''
        return self.get_page_num() == 0

    def table_fname(self):
        '''
        A function which formats names of pdf files with exam pages with added
        score tables.

        :name:
            Name of the pdf file with an exam page.
        '''
        return os.path.join(self.head,  "t_" + self.tail)

    def name_without_table(self):
        '''
        Given a name of a pdf file with an exam page with or without a score table
        returns the name of the corresponding pdf file without the table.
        '''
        if self.has_table():
            return os.path.join(self.head, self.tail[2:])
        else:
            return self.code

    def get_exam_name(self):
        '''
        Get exam base name (exam copy nummber and page number striped)
        '''
        return "-".join(self.get_exam_code().split("-")[:-1])

    

def covers_file(f):
    '''
    Given a name of a pdf file with with copies of an exam page, checks
    if it consists of cover pages
    '''
    return "problem_0.pdf" in os.path.basename(f)


def format_table(page, template, maxpoints=10):
    '''
    Formats a LaTeX template to add a score table to a given
    pdf page

    :page:
        Name of the pdf file to add score table to.
    :latex_template:
        Name of the LaTeX file with the table template.
    :maxpoints:
        The maximal number of points in the score table (up to 25 will look fine)
    Returns:
        A string with LaTeX code.
    '''


    # read the LaTeX template
    with open(template, 'r') as f:
        latex = f.read()


    # insert data into the template
    latex = (latex.replace("FILENAME", page)
             .replace("MAXPOINTS", str(maxpoints))
            )
    return latex



class PrepareGrading(GradingBase):

    def __init__(self, 
                 maxpoints, 
                 score_table_template, 
                 main_dir = None, 
                 gradebook = None, 
                 init_grading_data=False, 
                 show_pnums = False
                 ):
        
        GradingBase.__init__(self, main_dir, gradebook, init_grading_data)

        if type(maxpoints) != list:
            maxpoints = [maxpoints]
        self.maxpoints = maxpoints
        self.score_table_template = score_table_template        
        self.show_pnums = show_pnums

    def add_score_table(self):
        '''
        Addes score tables to pdf files with exam pages.

        :pages_dir:
            The directory with pdf files with exam pages.
        :template:
            LaTeX template used to add score tables.
        :maxpoints:
            Either an integer of a list with the maximal possible scores for
            each exam problem. This is used to format score tables. If maxpoints
            is an integer the score table for each problem will be limited to
            this number of points. If a list, the entries will be used to format
            score tables for consecutive problems. If the list contains fewer entries
            than there are problems, its last entry will be used for all remaining
            problems.
        '''

        # select pdf files with exam pages which do not have a score table
        files = glob.glob(os.path.join(self.pages_dir, "*.pdf"))
        files = [f for f in files if not ExamCode(f).has_table()]
        files.sort()

        max_score_dict = {}
        for f in files:

            fcode = ExamCode(f)
            output_file = fcode.table_fname()
            
            # if cover page, just copy it
            if fcode.is_cover():
                shutil.copy(f, os.path.join(self.pages_dir, output_file))
                continue

            page_num = fcode.get_page_num()
            max_score = self.maxpoints[min(page_num-1, len(self.maxpoints)-1)]
            max_score_dict[page_num] = max_score

            tex = format_table(page = f, template = self.score_table_template, maxpoints = max_score)
            c = compile_latex(tex, output_file = output_file , output_directory = self.pages_dir)
            if c[0] != 0:
                return f, c
            print(f"{os.path.basename(output_file)}   -->   {c[0]}\r", end="")
        print("Score tables added." + 40*" ")

        grading_data = self.get_grading_data()
        grading_data["maxpoints"] = max_score_dict
        self.set_grading_data(grading_data)


    def read_bubbles(self, img, dilx=(4,10), dily=(10, 4)):
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
        gray= cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]  #convert to binary
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
        if self.show_pnums:
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
        return str(person_number)


    def missing_qr_handler(self, scans, page_num, page_image, get_missing_data):
        if not get_missing_data:
            qr = None
            qr_found = False
            return qr, qr_found

        plt.figure(figsize = (15,20))
        plt.imshow(page_image)
        plt.show()

        msg = "\n\n"
        msg += f"File: {scans}\n"
        msg += f"Page: {page_num}\n"
        msg += "QR code not found. \n\n"
        msg += "Enter the exam code or 's' to skip for now: "
        qr = input(msg)
        qr = qr.strip()
        if qr == "s":
            qr_found = False
        else:
            qr_found = True
        return qr, qr_found


    def missing_pnum_handler(self, pnum, gradebook_df, scans, page_num, page_image, get_missing_data, show_page):
        if not get_missing_data:
            pnum = None
            pnum_found = False
            return pnum, pnum_found, gradebook_df
        
        if show_page:
            plt.figure(figsize = (15,20))
            plt.imshow(page_image)
            plt.show()
        
        pnum_found = False
        while not  pnum_found:
            msg = "\n\n"
            msg += f"File: {scans}\n"
            msg += f"Page: {page_num}\n"
            msg += f"Person number has been read as: {pnum}.\n"
            msg += "This person number is not recognized.\n\n"
            msg += f"Enter person number, or 'add' to add {pnum} to the gradebook, or 's' to skip for now: "
            new_pnum = input(msg)
            new_pnum = new_pnum.strip()
            if new_pnum == "s":
                pnum = None
                pnum_found = False
                break
            elif new_pnum == "add":
                new_row = {self.pnum_column:[pnum]}
                new_row_df = pd.DataFrame(new_row)
                gradebook_df = gradebook_df.append(new_row_df, sort=False).reset_index().drop(columns = ["index"])
                pnum_found = True
                break
            else:
                pnum = new_pnum
                pnum_found = (pnum in gradebook_df[self.pnum_column].values)

        return pnum, pnum_found, gradebook_df
                


    def read_scans(self, scans, get_missing_data=False):
        '''
        Given a pdf file with scanned exams:
            - reads the QR code from each page
            - if the page is an exam cover page reads the person number
            - writes the exam code associated to the person number in the gradebook
            - saves each scanned page as an individual pdf file; the name of this file
            if the QR code of the page.

        :scans:
            The name of the pdf file to be processed.
        :gradebook:
            The csv file containing student person numbers.
        :pages_dir:
            The directory where pdfs with individual exam pages will be saved.
        :show_pnums:
            An argument of the read_bubbles function. If True, displayes images
            showing how person numbers are read.

        Returns:
            None
        '''

        if not os.path.exists(self.pages_dir):
            os.makedirs(self.pages_dir)
        if not os.path.exists(self.for_grading_dir):
            os.makedirs(self.for_grading_dir)

        processing_missing_data_file = ((os.path.realpath(scans) ==  os.path.realpath(self.missing_data_pages)))

        # read gradebook, add qr_code column if needed
        gradebook_df = pd.read_csv(self.gradebook, converters={self.pnum_column : str})
        if self.qr_code_column not in gradebook_df.columns:
            gradebook_df[self.qr_code_column] = ""

    

        # writer object for collecting pages with missing data
        missing_data_writer = pdf.PdfFileWriter()
        
        had_missing_file = os.path.isfile(self.missing_data_pages)
        missing_data = []

        if not processing_missing_data_file:
            missing_data = self.get_grading_data()["missing_data"]
            # if a file with missing data pages exists, copy its 
            # content to missing_data_writer 
            if had_missing_file:
                missing_data_file = open(self.missing_data_pages, 'rb')
                missing_data_pdf = pdf.PdfFileReader(missing_data_file)
                missing_data_writer.appendPagesFromReader(missing_data_pdf)  
            
        # read scans
        with open(scans, 'rb') as f:
            scanned_pdf = pdf.PdfFileReader(f)
            num_pages = scanned_pdf.numPages

            # for each page of the pdf file:
            for n in range(num_pages):

                # flag indicating if the exam page is to be displayed
                # if data is missing
                show_page = True

                # convert the page into a PIL image object
                page = pdf.PdfFileWriter()
                page.addPage(scanned_pdf.getPage(n))
                page_image = pdfpage2img(page)

                # get QR code from the page
                qr_list = enhanced_qr_decode(page_image)

                # check is a QR code was found on the page
                # if not found, call missing_qr_handler
                qr_found = (len(qr_list) != 0)
                if qr_found:
                    qr = qr_list[0].data.decode('utf8')
                else: 
                    qr, qr_found = self.missing_qr_handler(scans = scans, 
                                                            page_num = n+1, 
                                                            page_image = page_image, 
                                                            get_missing_data  = get_missing_data
                                                            )
                    show_page = False
                    # if qr code data is still missing, add the page to the missing data
                    # file if meeded and go to the next page
                    if not qr_found:
                        if [os.path.basename(scans), n] not in missing_data:
                            missing_data_writer.addPage(scanned_pdf.getPage(n))
                            missing_data.append([os.path.basename(scans), n])
                        continue
                
                # if qr code was found get person number (if cover page)
                qr_code = ExamCode(qr)
                # check if cover page, if so read the person number
                # and record the QR code in the gradebook
                if qr_code.is_cover():
                    # read the person number
                    pnum = self.read_bubbles(page_image)
                    # check if the person number read is in the gradebook
                    pnum_found = (pnum in gradebook_df[self.pnum_column].values)
                
                    if not pnum_found:
                        pnum, pnum_found, gradebook_df = self.missing_pnum_handler(pnum = pnum, 
                                                                                    gradebook_df = gradebook_df, 
                                                                                    scans = scans, 
                                                                                    page_num = n+1, 
                                                                                    page_image = page_image, 
                                                                                    get_missing_data = get_missing_data, 
                                                                                    show_page = show_page
                                                                                    )
                        if not pnum_found:
                            if [os.path.basename(scans), n] not in missing_data:
                                missing_data_writer.addPage(scanned_pdf.getPage(n))
                                missing_data.append([os.path.basename(scans), n])
                            continue

                    print(f"person_number: {pnum}\r", end="")
                    # find the number of the row in the dataframe with the given person number
                    i = np.flatnonzero(gradebook_df[self.pnum_column].values == pnum)[0]
                    # record the  QR code of a student exam in the gradebook
                    gradebook_df.loc[i, self.qr_code_column] = qr_code.get_exam_code()

                # save the exam page as a pdf file, the file name is the QR code of the page
                page_file = os.path.join(self.pages_dir, qr + ".pdf")
                with open(page_file , 'wb') as f:
                    page.write(f)
                    print(qr + "\r", end="")

        # save the missing pages file or remove if empty
        if missing_data_writer.getNumPages() > 0:
            temp_file = self.missing_data_pages + "_temp"
            with open(temp_file, 'wb') as f:
                missing_data_writer.write(f)
            if had_missing_file:
                if not processing_missing_data_file:
                    missing_data_file.close()
                os.remove(self.missing_data_pages)
            os.rename(temp_file, self.missing_data_pages)
            grading_data = self.get_grading_data()
            grading_data["missing_data"] = missing_data
            self.set_grading_data(grading_data)
            
        else:
            if had_missing_file:
                os.remove(self.missing_data_pages)
            grading_data = self.get_grading_data()
            grading_data["missing_data"] = []
            self.set_grading_data(grading_data)

        # save the gradebook
        gradebook_df.to_csv(self.gradebook, index=False)



    def assemble_by_problem(self):
        '''
        Assembles pages of exam copies into files, one file containing
        all copies of a given page. Pages within each file are sorted
        according to their QR codes.
        For each pdf file it creates, this function also writes a json file
        with a list of pages (in order) that went into that file.

        :pages_dir:
            The directory containing pdf files with exam pages. Json files will
            be saved in this directory.
        :grading_dir:
            The directory where the assembled pdf files are to be saved.

        Returns:
            None.
        '''


        files = glob.glob(os.path.join(self.pages_dir, "*.pdf"))
        # list of pdf file with score tables
        files = [f for f in files if ExamCode(f).has_table()]

        files_dir = {}
        for f in files:
            fcode = ExamCode(f)
            # get the page number
            files_dir[f] = fcode.get_page_num()

        # create the set of page (or problem) numbers of the exam
        problems = set(files_dir.values())
        
        page_lists = {}
        for n in problems:
            # list of pages with the problem n, sorted by QR codes
            f_n = [f for f in files_dir if files_dir[f] == n]
            f_n.sort()

            exam_name = ExamCode(f_n[0]).get_exam_name()
            output = f"{exam_name}_problem_{n}"

            output_fname = os.path.join(self.for_grading_dir, output + ".pdf")
            merge_pdfs(f_n , output_fname=output_fname)

            page_lists[os.path.basename(output_fname)] = [os.path.basename(f) for f in f_n]
        
        grading_data = self.get_grading_data()
        grading_data["page_lists"] = page_lists
        self.set_grading_data(grading_data)

 


    def prepare_grading(self, files=None):

        '''
        Prepares exams for grading:
        - It creates directories "pages", "for_grading", and "graded" in
            the main grading directory (if they don't exists yet).
        - It reads scanned pdf files in the "scans" directory using the read_scans
            function.
        - It adds a score table to each page of the exam directory using the add_score_table
            function.
        - It assembles problem for grading and places them in the "for_grading" directory
            using the assemble_by_problem function.


        :main_dir:
            The main grading directory. If none, the currect directory will be used.
        :gradebook:
            The csv file containing student person numbers.
        :grade_table_template:
            LaTeX template used to add score tables.
        :maxpoints:
            Either an integer of a list with the maximal possible scores for
            each exam problem. This is used to format score tables. If maxpoints
            is an integer the score table for each problem will be limited to
            this number of points. If a list, the entries will be used to format
            score tables for consecutive problems. If the list contains fewer entries
            than there are problems, its last entry will be used for all remaining
            problems.
        :show_pnums:
            An argument of the read_bubbles function. If True, displayes images
            showing how person numbers are read.

        Returns:
            None
        '''

        if not os.path.exists(self.pages_dir):
            os.makedirs(self.pages_dir)

        processed_scans = self.get_grading_data()["processed_scans"]
        processed_scans.append(os.path.basename(self.missing_data_pages))
        processed_scans_set = set(processed_scans)

        if files is None:
            file_list = set([os.path.basename(f) for f in set(glob.glob(os.path.join(self.scans_dir, "*.pdf")))])
            file_list = list(file_list.difference(processed_scans_set))
        elif files == "all":
            file_list = [os.path.basename(f) for f in set(glob.glob(os.path.join(self.scans_dir, "*.pdf")))]
            processed_scans_set = set()
        elif type(files) == list:
            file_list = [os.path.basename(f) for f in files]
        
        file_list.sort()

        print("\nReading scanned files... \n")
        processed = []
        for f in file_list:
            print(f"Reading file:  {f}")
            fpath = os.path.join(self.scans_dir, f)
            if not os.path.exists(fpath):
                print(f"File {f} not found, omitting.")
                continue
            self.read_scans(scans = fpath, get_missing_data=False)
            processed.append(f)
        
        if  os.path.isfile(self.missing_data_pages):
            print(f"Reading file:  {os.path.basename(self.missing_data_pages)}\n")
            self.read_scans(scans = self.missing_data_pages, get_missing_data=True)
        

        print("\n\nAdding score tables... \n")
        self.add_score_table()

        self.split_for_grading_files(dest_dir = self.pages_dir)

        print("\n\nAssembling files for grading... \n")
        self.assemble_by_problem()

        print("\n\nFinishing...")
        grading_data = self.get_grading_data()
        grading_data["processed_scans"] = list(set(processed_scans_set).union(processed))
        self.set_grading_data(grading_data)
        shutil.rmtree(self.pages_dir)
        print("\nGrading files ready.")






















def read_problem_scores(fname, maxpoints, treshold = 250):
    '''
    Reads scores from a pdf file of problems with score tables.
    It assumes that the file consists of copies of the same problem,
    which have the same maximal point value.

    :fname:
        The name of the pdf file.
    :maxpoints:
        The maximum point value of the graded problems.
    :treshold:
        Integer value for detecting if a box of the score table is checked
        or not each score box is white, so when it us unmarked the mean of its
        pixel values is 255. If the mean read from the pdf is below the treshhold
        we count the score box as marked

    Returns:
        A list of scores, one for each pdf page. If no marked score boxes are detected
        the value of the list for the page will be "NONE". If multiple marked boxes are
        detected, the value of the list for the page will be "MULTI" followed by the list
        of the read scores.
    '''

    pages = pdf2image.pdf2image.convert_from_path(fname)
    # row and column offset of the first score box in the score table
    row = 2104
    col = 61

    #list of scores
    scores = []
    for page in pages:
        img = np.array(page)
        score_table = []
        # iterate over score boxes in the score table
        for i in range(maxpoints+1):
            x = col + i*59
            box = img[row:row+32, x:x+32, :]
            if box.mean() < treshold:
                score_table.append(i)

        if len(score_table) == 1:
            scores.append(score_table[0])
        elif len(score_table) == 0:
            scores.append("NONE")
        else:
            scores.append("MULTI: " + str(score_table))
    return scores


def get_scores_df(main_dir=None):
    '''
    Reads scores from graded pdf files with exam problems.

    :main_dir:
        The main grading directory. If none, the currect directory will be used.

    Returns:
        Pandas dataframe with exam scores. Rows are indexed with exam codes,
        columns with problem numbers. A "NONE" value in the dataframe indicates
        that no score has been detected. A "MULTI" value indicates that multiple
        scores for the same problem have been detected and gives the list of these
        values.
    '''

    global pages_dir
    global for_grading_dir

    if main_dir is None:
        main_dir = os.getcwd()

    for_grading_dir = os.path.join(main_dir, for_grading_dir)
    pages_dir = os.path.join(main_dir, pages_dir)


    # get files with exam problems, skip the file with exam covers
    files = glob.glob(os.path.join(for_grading_dir, "*_problem_*.pdf"))
    files = [f for f in files if not covers_file(f)]

    # load the list with max score for each problem
    with open(os.path.join(pages_dir, "max_points.json")) as foo:
        maxpoint_list = json.load(foo)

    score_dict = {}
    for fname in files:
        print(f"Processing: {fname}\r", end="")

        basename = os.path.splitext(os.path.basename(fname))[0]
        #json file with qr codes of pages in the pdf file
        json_fname =  os.path.join(pages_dir, basename + ".json")
        page_num = int(basename.split("_")[-1])
        # maximal possible score for the problem
        maxpoints = maxpoint_list[str(page_num)]

        score_list = read_problem_scores(fname=fname, maxpoints=maxpoints)

        # associate problem  scores with exam codes
        with open(json_fname) as foo:
            pages = json.load(foo)
        pages = [get_exam_code(p) for p in pages]
        if len(pages) != len(score_list):
            return None
        score_dict_page = {p:s for (p,s) in zip(pages, score_list)}
        score_dict["prob_" + str(page_num)] = score_dict_page

    scores_df = pd.DataFrame(score_dict)
    return scores_df



def get_scores(main_dir = None, gradebook = None, save = False, new_gradebook = None):

    global qr_code_column
    global total_column
    global grade_column
    global default_gradebook
    global for_grading_dir

    if main_dir is None:
        main_dir = os.getcwd()

    if gradebook is None:
        gradebook = os.path.join(main_dir, default_gradebook)
    if new_gradebook is None:
        new_gradebook = gradebook

    for_grading_dir = os.path.join(main_dir, for_grading_dir)

    scores_df = get_scores_df(main_dir = main_dir)

    problem_cols = scores_df.columns.tolist()
    scores_df[total_column] = scores_df[problem_cols].sum(axis=1)

    gradebook_df = pd.read_csv(gradebook)

    for col in problem_cols + [total_column]:
        try:
            gradebook_df.drop(columns = col, inplace=True)
        except KeyError:
            continue

    new_gradebook_df = pd.merge(gradebook_df, scores_df ,how = "left", right_index = True, left_on = qr_code_column)
    new_gradebook_df[total_column] = new_gradebook_df[problem_cols].sum(axis=1)
    if grade_column not in new_gradebook_df.columns:
        new_gradebook_df[grade_column] = ""

    if save:
        new_gradebook_df.to_csv(new_gradebook, index=False)

    return scores_df, new_gradebook_df


def cover_page_grades(page, scores, total, grade, template, output_file, output_directory, extras = None):


    if extras is None:
        extras = {}

    def format_score(v):
        try:
            score = int(v)
        except:
            score = "--"
        return score

    boxnum = len(scores) + len(extras) + 2
    scores_str = []
    for i, s in enumerate(scores):
        scores_str.append(f"{i+1}/{format_score(s)}/{i+1}")
    scores_str = ",".join(scores_str)

    i = len(scores)
    for k in extras:
        scores_str += f",{i+1}/{format_score(extras[k])}/{k}"
        i +=1

    grade_str = f"{boxnum-1}/{format_score(total)}/TOTAL, {boxnum}/{grade}/GRADE"

    # read the LaTeX template
    with open(template, 'r') as f:
        latex = f.read()

    # insert data into the template
    latex = latex.replace("FILENAME", page)
    latex = latex.replace("BOXNUM", str(boxnum))
    latex = latex.replace("SCORES", scores_str)
    latex = latex.replace("GRADE", grade_str)

    c = compile_latex(latex, output_file = output_file , output_directory = output_directory)

    return c


def flatten_pdf(fname):
    ps_fname = os.path.splitext(fname)[0] + ".ps"
    c1 = f'pdftops "{fname}"  "{ps_fname}"'
    c2 = f'ps2pdf "{ps_fname}" "{fname}"'
    os.system(c1)
    os.system(c2)
    os.remove(ps_fname)





def assemble_by_student(main_dir = None, gradebook = None, extras = None, flatten = False):

    global qr_code_column
    global total_column
    global grade_column
    global pnum_column
    global for_grading_dir
    global pages_dir
    global graded_dir
    global default_gradebook


    if main_dir is None:
        main_dir = os.getcwd()
    if gradebook is None:
        gradebook = os.path.join(main_dir, default_gradebook)

    full_pages_dir = os.path.join(main_dir, pages_dir)
    full_for_grading_dir = os.path.join(main_dir, for_grading_dir)
    full_dest_dir = os.path.join(main_dir, graded_dir)

    gradebook_df =  pd.read_csv(gradebook)
    files = glob.glob(os.path.join(for_grading_dir, "*_problem_*.pdf"))

    temp_dir = tempfile.mkdtemp()

    split_for_grading(main_dir = main_dir, dest_dir = temp_dir)

    covers = sorted([f for f in glob.glob(os.path.join(temp_dir, "*.pdf")) if cover_page(get_fname(f))])
    prob_cols = sorted([c for c in gradebook_df.columns.tolist() if "prob_" in c])
    for cover in covers:
        basename = os.path.splitext(os.path.basename(cover))[0]
        print(f"{basename}\r", end="")
        cover_copy = os.path.join(temp_dir, basename + "_copy.pdf")
        shutil.copyfile(cover, cover_copy)
        qr = get_exam_code(basename)
        record = gradebook_df.loc[gradebook_df[qr_code_column] == qr]
        scores = record[prob_cols].values[0]
        total = int(record[total_column].values[0])
        grade = str(record[grade_column].values[0])
        extra_vals = {}
        for k in extras:
            extra_vals[k] =  record[extras[k]].values[0]
        c, d = cover_page_grades(page=cover_copy,
                                 scores=scores,
                                 total=total,
                                 grade=grade,
                                 extras = extra_vals,
                                 template = "cover_page_grades_template.tex",
                                 output_file=basename,
                                 output_directory = temp_dir)
        os.remove(cover_copy)


    files = glob.glob(os.path.join(temp_dir, "*.pdf"))
    codes = set( get_exam_code(get_fname(f)) for f in files)
    for exam_code in codes:
        exam_pages = [f for f in files if get_exam_code(get_fname(f)) == exam_code]
        exam_pages.sort()
        output_fname = os.path.join(dest_dir, exam_code + ".pdf")
        merge_pdfs(exam_pages, output_fname=output_fname)

    exam_files = glob.glob(os.path.join(dest_dir, "*.pdf"))

    if flatten:
        for f in exam_files:
            flatten_pdf(f)

    shutil.rmtree(temp_dir)
