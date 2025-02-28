\ProvidesClass{thesis}[]

% Layout
\LoadClass[a4paper,11pt,twoside,english]{book}
\usepackage[a4paper,left=3cm,right=3cm,bottom=3cm,top=3cm]{geometry}
\renewcommand{\baselinestretch}{1.25}
\pagestyle{plain}
\usepackage{emptypage}
\setlength\parindent{0pt}
\setlength{\columnsep}{20pt}

% Base packages
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage[dvipsnames]{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{wrapfig}
\usepackage{graphicx}
\usepackage[labelfont=bf]{caption}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{mathtools}

% Graphics
\usepackage{tikz}
\usepackage{tikz-3dplot}
\usetikzlibrary{decorations.pathreplacing}
\usetikzlibrary{arrows.meta}
\usetikzlibrary{intersections, positioning}
% \usepackage{pgfplots}

% Bibliography
\usepackage{csquotes}
\usepackage{biblatex}
\addbibresource{references.bib}

% Math shorthands
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}

\newcommand{\unormal}{\mathcal{N}(\mathbf{0},\mathbf{I})}
\newcommand{\normal}[2]{\mathcal{N}(#1,#2)}
\newcommand{\fnormal}[3]{\mathcal{N}(#1;#2,#3)}

\newcommand{\E}[1]{\mathbb{E} [#1]}
\DeclareMathOperator{\Var}{Var}

\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}

\DeclarePairedDelimiterX{\norm}[1]{\lVert}{\rVert}{#1}

% Avoid widows and orphans
\clubpenalty = 10000
\widowpenalty = 10000 
\displaywidowpenalty = 10000

% Avoid inline math breaking
\relpenalty=10000
\binoppenalty=10000
