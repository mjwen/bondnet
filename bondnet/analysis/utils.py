import numpy as np
from beautifultable import BeautifulTable


class TexWriter:
    @staticmethod
    def head():
        s = r"""
\documentclass[11pt]{article}

\usepackage[top=1in, bottom=1in, left=1in, right=1in]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{color}
\usepackage{float}

\begin{document}
"""
        return s

    @staticmethod
    def tail():
        return r"\end{document}"

    @staticmethod
    def newpage():
        return "\n" * 2 + r"\newpage" + "\n"

    def verbatim(s):
        return "\n" + r"\begin{verbatim}" + "\n" + s + "\n" + r"\end{verbatim}" + "\n"

    @staticmethod
    def single_figure(filename, figure_size=0.4):
        s = r"""
\begin{figure}[H]
\centering
\includegraphics[width=%s\columnwidth]{%s}
\end{figure}
""" % (
            str(figure_size),
            filename,
        )
        return s

    def resize_string(s, length=80, start="", end="\n"):
        """
        Reshape a string `s` to substrings with `start` prepended and `end` appended at
        each substring.

        Returns: a string
        """
        s = [start + s[0 + i : length + i] + end for i in range(0, len(s), length)]
        return "".join(s)

    @staticmethod
    def beautifultable(
        array,
        header,
        first_column=None,
        first_column_header=None,
        num_tables=1,
        to_string=True,
    ):
        """
        Convert a 2D array to a beautiful table, with the ability to separate the table
        into multiple ones in case there are too many columns.

        first_column (1D array): additional data (e.g. index) to be added to the first
            column of each table.
        first_column_header (str): header for the first column
        num_tabels (int): number of tables to split the array (along column)
        to_string (bool): if `True` convert the tables to a string.

        Returns:
            a list of beautiful tabels or a stirng of `to_string` is `True`
        """
        array = np.asarray(array)

        tables = []
        headers = []
        arrays = []
        column_width = int(np.ceil(len(array[0]) / num_tables))
        for i in range(num_tables):
            tables.append(BeautifulTable(max_width=80))

            if first_column_header is not None:
                headers.append(
                    [first_column_header]
                    + header[i * column_width : (i + 1) * column_width]
                )
            else:
                headers.append(header[i * column_width : (i + 1) * column_width])

            a = array[:, i * column_width : (i + 1) * column_width]
            if first_column is not None:
                fc = np.atleast_2d(first_column).T
                a = np.concatenate((fc, a), axis=1)
            arrays.append(a)

        for t, h, a in zip(tables, headers, arrays):
            t.column_headers = h
            for row in a:
                t.append_row(row)

        res = ""
        if to_string:
            for table in tables:
                for line in str(table):
                    res += line
                res += "\n"
            tables = res

        return tables
