We provide code to read filtrations. You can download it either in C++ or in Java. You can otherwise reproduce it in any other language of your choice.

This code assumes that the filtration is given in an ASCII file with the following format, where each line represents a simplex sigma and is of the form:

f(sigma) dim(sigma) v_0 ... v_{dim(sigma)}

where:

f(sigma) is the function value of sigma (its "time of appearance" in the filtration),
dim(sigma) is the dimension of sigma,
v_0 ... v_{dim(sigma)} are the IDs (integers) of the vertices of sigma.
For instance, 0.125 2 2 6 4 denotes a 2-simplex (triangle) that appears in the filtration at time 0.125 and whose vertices have IDs 2, 6 and 4. Warning: the function values provided in the file must be compatible with the underlying simplicial complex (function values of simplices must be at least as large as the ones of their faces). Nevertheless, the vertex IDs are arbitrary integers and may not start at 0 nor be continuous.

The code produces a vector of simplices F, each simplex being described as a structure with fields val (float), dim (integer) and vert (the sorted set of the vertex IDs of the simplex). Warning: the order of the simplices in F may not be the one of the filtration.
