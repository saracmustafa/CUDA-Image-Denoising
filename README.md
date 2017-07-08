# CUDA-Image-Denoising

:computer: &nbsp;**Fall 2016 COMP 429 Parallel Programming Course Project, Koç University**

The final version of this code has been developed by **Mustafa SARAÇ** and **Mustafa Mert ÖGETÜRK** as a project of Parallel Programming (COMP 429) course. **Koç University's code of ethics can be applied to this code and liability can not be accepted for any negative situation. Therefore, be careful when you get content from here.**

**Description:** The serialized version of the noise removal algorithm was parallelized using various methods via the CUDA library.
 
The parallelization process consists of **3 steps** in total and these are as follows:
 - noise_remover_v1.cu (**Naive implementation**)
 - noise_remover_v2.cu (**Using Temporary Variables to Eliminate Global Memory References**) 
 - noise_remover_v3.cu (**Using Shared Memory on the GPU**)
 
You can also view our **project report**.

#### For more detailed questions, you can contact me at this email address: msarac13@ku.edu.tr &nbsp;&nbsp;:email:
