"""Tasks for compiling the paper and presentation(s)."""
import shutil

import pytask
from frechet_fda.config import BLD, PAPER_DIR
from pytask_latex import compilation_steps as cs

documents = ["frechet_fda", "frechet_fda_pres"]

DEPENDENCIES = {
    "Chapter 1": PAPER_DIR / "Introduction.tex",
    "Chapter 2": PAPER_DIR / "Maths.tex",
    "Chapter 3": PAPER_DIR / "FDA.tex",
    "Chapter 4": PAPER_DIR / "Densities.tex",
    "Chapter 5": PAPER_DIR / "Fréchet.tex",
    "Chapter 6": PAPER_DIR / "Application.tex",
    "Chapter 7": PAPER_DIR / "Conclusion.tex",
    "Appendix A": PAPER_DIR / "Numerics.tex",
    "Appendix B": PAPER_DIR / "Code.tex",
    "Appendix C": PAPER_DIR / "Figures.tex",
},

for document in documents:

    @pytask.mark.latex(
        script=PAPER_DIR / f"{document}.tex",
        document=BLD / "latex" / f"{document}.pdf",
        compilation_steps=cs.latexmk(
            options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
        ),
    )
    @pytask.task(id=document)
    def task_compile_document(paths = DEPENDENCIES):
        """Compile the document specified in the latex decorator."""

    kwargs = {
        "depends_on": BLD / "latex" / f"{document}.pdf",
        "produces": BLD.parent.resolve() / f"{document}.pdf",
    }

    @pytask.task(id=document, kwargs=kwargs)
    def task_copy_to_root(depends_on, produces):
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)
