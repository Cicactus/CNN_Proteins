This is the program to perform protein both local (by Protein Blocks) and secondary structures prediction
by precomputed learned NN model without reliancy to the evolutionary information (i.e. sequence alignments).
(C) Yuri V. Milchevskiy, Yuri V. Kravatsky under the terms of the Creative Commons CC BY-NC-SA 4.0 license
email: milch@eimb.ru

1. Unpack predict_common.tar.gz:
   tar -xf predict_common.tar.gz

2. Download databases:
   wget https://ftp.eimb.ru/Milch/Predict_byNN/predict_dbs.tar.bz2
   or
   wget ftp://ftp.eimb.ru/Milch/Predict_byNN/predict_dbs.tar.bz2

3. Put predict_dbs.tar.bz2 to directory "databases" and unpack it:
   tar -xf predict_dbs.tar.bz2

4. Unpack binary for your OS or get and compile predict_byNN.tgz:
   tar -xf predict_byNN.tgz
   cd predict
   make -j8 makefile

For Windows you should compile static release under MSYS2:
   make -j8 makefile.win

Now you can get release for your OS in bin/Release directory.
The compile process is tested with g++ v. 7, 9, 10, 11, clang12.

5. Install python3 and all proper dependencies:
   1. python 3 (tested with v. 3.8.10)
   2. numpy (tested with v. 1.22.3)
   3. pandas (tested with v. 1.5.1)
   4. PyTorch (tested with v. 1.12.1)

6. Run protein structure prediction by the following way:
   a. predict_byNN -cALPPPGGGGLSLSLSLS
   You can get protein stuctures prediction for the sequence from the command string.
   Result will be stored in the file "Sequence_setted_in_CommandLine"

   b. predict_byNN -f1A0TP.fasta
   In this way you can get protein structures prediction for the sequence from the FASTA file

   c. predict_byNN  -s1A0TP.seq
   In this way you can get protein stuctures prediction for the sequence from the seq file 
   (no header, just sequence)
