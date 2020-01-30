import numpy as np
from beautifultable import BeautifulTable
from collections import defaultdict


def graph_struct_list_representation(g):
    """
    Represent the heterogrpah canonical edges as a dict.

    Args:
        g: a dgl heterograph

    Returns:
        dict of dict of list: dest node is the outer key; `nodes`and `edges` are the
        innter keys; src nodes are the values associated with `nodes` and edge types
        are the vlues associated with `edges`.

    Example:
        Suppose the graph has canonical edges:
         src   edge   dest

        [['A', 'A2B', 'B'],
         ['C', 'C2B', 'B'],
         ['C', 'C2A', 'A']]

        This function rturns:
        {
         'A':{'nodes':['C'],
              'edges':['C2A']},
         'B':{'nodes':['A', 'C'],
              'edges':['A2B', 'C2B']}
        }
    """
    graph_strcut = {
        t: {"nodes": defaultdict(list), "edges": defaultdict(list)} for t in g.ntypes
    }
    for src, edge, dest in g.canonical_etypes:
        graph_strcut[dest]["nodes"].append(src)
        graph_strcut[dest]["edges"].append(edge)

    return graph_strcut


def get_bond_to_atom_map(g):
    """
    Query which atoms are associated with the bonds.

    Args:
        g: dgl heterograph

    Returns:
        dict: with bond index as the key and a tuple of atom indices of atoms that
            form the bond.
    """
    nbonds = g.number_of_nodes("bond")
    bond_to_atom_map = dict()
    for i in range(nbonds):
        atoms = g.successors(i, "b2a")
        bond_to_atom_map[i] = sorted(atoms)
    return bond_to_atom_map


def get_atom_to_bond_map(g):
    """
    Query which bonds are associated with the atoms.

    Args:
        g: dgl heterograph

    Returns:
        dict: with atom index as the key and a list of indices of bonds is
        connected to the atom.
    """
    natoms = g.number_of_nodes("atom")
    atom_to_bond_map = dict()
    for i in range(natoms):
        bonds = g.successors(i, "a2b")
        atom_to_bond_map[i] = sorted(list(bonds))
    return atom_to_bond_map


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
