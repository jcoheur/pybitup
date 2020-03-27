def style():
    fontSize = 36
    params = {
		'font.family':'serif',
		'font.serif':'Computer Modern',
        'font.size': fontSize,
        'axes.labelsize': fontSize,
        'legend.fontsize': fontSize-5,
        'xtick.labelsize': fontSize-5,
        'ytick.labelsize': fontSize-5,
        'axes.grid': False,
        'xtick.top': False,
        'ytick.right': False,
        'figure.figsize': [16, 9],
        'text.usetex': True,
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
            r"\usepackage[T1]{fontenc}",  # plots will be generated
            r"\usepackage[version=4]{mhchem}",
            r"\usepackage{siunitx}",
        ],
        'axes.linewidth': 1.5,
        "xtick.minor.visible": False,
        "lines.linewidth":3,
        "lines.markersize": 7,
        "lines.markeredgewidth": 2.0,
        "legend.frameon": False,
        "legend.edgecolor":(1,1,1),
        "legend.framealpha": 0.5,
        "legend.fancybox": "False",
        "xtick.major.size":10,
        "xtick.major.width":2.5,
        "xtick.minor.size": 6,
        "xtick.minor.width": 1.5,
        "ytick.major.size":10,
        "ytick.major.width":2.5,
        "ytick.minor.size": 6,
        "ytick.minor.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
    return params