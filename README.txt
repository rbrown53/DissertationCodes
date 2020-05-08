

This collection of code contains everything needed to reproduce the results contained in my PhD dissertation, ``SEMIVARIOGRAM METHODS FOR MODELING WHITTLE-MATÉRN PRIORS IN BAYESIAN INVERSE PROBLEMS." Alternatively, you may use the code to use the methods described in the dissertation to solve your own inverse problems. All code was either written by Rick Brown or the author is given at the top of the file. There are four main files: Inpainting.m, Inpainting_MCMC.m, Inpainting_Regional_Sparse.m, and Inpainting_Regional_Mult.m. Various functions needed to implement all of the methods contained in these files are in the `Functions&MatFiles' folder. Each function is described in its respective file. Additionally, the images for Main Hall and the Wave are contained in the `Example_Images' folder.


Inpainting.m can be used to implement inpainting in the isotropic or anisotropic case when only a single prior is used for the entire image. When running the file, the user will be asked whether to use a black and white image (this will also convert a color image to grayscale) or a color image and what type of prior to use. There are four types of priors: an isotropic Whittle-Matérn prior, an anisotropic Whittle-Matérn prior, a Tikhonov prior, or a Laplacian prior. By default, the Main Hall image will be read in. Adjustments to the code can be made to read in any image.


Inpainting_MCMC.m can be used to implement Markov Chain Monte Carlo methods for estimating, and quantifying uncertainty in, hyperparameters of a Whittle-Matérn prior for an inpainting problem. This can also be done in grayscale or color. 


Inpainting_Regional_Sparse.m implements the method for dealing with images with regions that need different priors that is described in Section 5.1 of the dissertation. The user will draw the distinct regions of the image using their cursor and a different prior will be used for each region. DFTs are used when they can be, but this implementation also relies on using sparse precision matrices to obtain the results.


Inpainting_Regional_Mult.m is based on an alternative to Inpainting_Regional_Sparse.m that is suggested in Section 5.1.8 of the dissertation. Instead of solving one inverse problem, the number of inverse problems solved is equal to the number of distinct regions of the image. Each one is solved independently and the results are then stitched together. This method was not used in the dissertation.  

Questions can be sent to rick.brown@umontana.edu.
