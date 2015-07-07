Authorship attribution methods excel under the assumption that the true author of a text is already in the list of suspects. However, when performing attribution "in the wild," researchers often cannot make that assumption. This script aims to produce a naive measurement of confidence that an unknown text belongs to a suspected author, by sampling from the suspect's writing and approximating its degrees and features of internal variation. The script is designed with literary applications in mind.

The script relies on standard scientific Python packages: pandas, scikit-learn, & matplotlib. It also relies on the Natural Language Toolkit, for tokenization.

For an example and some parameters of the Smell Test, see: http://teddyroland.com/2015/07/02/attributing-authorship-to-iterating-grace-or-the-smell-test-of-style/
