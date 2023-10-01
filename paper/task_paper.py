"""Tasks for compiling the paper and presentation(s)."""
import shutil

import pytask
from frechet_fda.config import BLD, PAPER_DIR
from pytask_latex import compilation_steps as cs

documents = ["frechet_fda"]

for document in documents:

    @pytask.mark.latex(
        script=PAPER_DIR / f"{document}.tex",
        document=BLD / "latex" / f"{document}.pdf",
        compilation_steps=cs.latexmk(
            options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
        ),
    )
    @pytask.mark.task(id=document)
    @pytask.mark.depends_on(
        {
            "Chapter 1": PAPER_DIR / "1_Introduction.tex",
            "Chapter 2": PAPER_DIR / "2_FDA.tex",
            "Chapter 3": PAPER_DIR / "3_Densities.tex",
            "Chapter 4": PAPER_DIR / "4_Fréchet.tex",
            "Chapter 5": PAPER_DIR / "5_Simulation.tex",
            "Chapter 6": PAPER_DIR / "6_Conclusion.tex",
            "Appendix A": PAPER_DIR / "A_NPRegression.tex",
            "Appendix B": PAPER_DIR / "B_Numerics.tex",
        },
    )
    def task_compile_document():
        """Compile the document specified in the latex decorator."""

    kwargs = {
        "depends_on": BLD / "latex" / f"{document}.pdf",
        "produces": BLD.parent.resolve() / f"{document}.pdf",
    }

    @pytask.mark.task(id=document, kwargs=kwargs)
    def task_copy_to_root(depends_on, produces):
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)
