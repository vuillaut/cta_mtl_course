import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os



def test_notebooks():
    indir = '.'
    all_notebooks = [os.path.join(indir, f) for f in os.listdir(indir) if f.endswith('.ipynb')]

    for notebook_filename in all_notebooks:
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': indir}})
