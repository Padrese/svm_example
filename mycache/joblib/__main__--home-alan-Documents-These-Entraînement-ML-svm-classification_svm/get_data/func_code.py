# first line: 15
@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return sp.sparse.lil_matrix(data[0]).toarray(), np.array(data[1]) #Data parsing from Scipy's sparse matrix to NumPy array format
