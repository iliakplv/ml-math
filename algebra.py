def list_mul(l1, l2):
    return [l1[i] * l2[i] for i in range(len(l1))]


def list_sum(l):
    sum = 0
    for item in l:
        sum += item
    return sum


class Vector:
    def __init__(self, vector):
        if not isinstance(vector, list):
            raise Exception('Vector is not a list')
        self.vector = vector

    def __len__(self):
        return len(self.vector)

    def __add__(self, other):
        if len(self) != len(other):
            raise Exception('Different size vectors')
        return Vector([self.vector[i] + other.vector[i] for i in range(len(self))])

    def __mul__(self, other):
        return list_sum(list_mul(self.vector, other.vector))

    def print(self):
        for item in self.vector:
            print('[ {} ]'.format(item))


class Matrix:
    def __init__(self, matrix):
        if not isinstance(matrix, list):
            raise Exception('Matrix is not a list')
        self.rows = len(matrix)
        self.cols = None
        for row in matrix:
            if not isinstance(row, list):
                raise Exception('Matrix row is not a list')
            if self.cols is None:
                self.cols = len(row)
            elif self.cols != len(row):
                raise Exception('Matrix has uneven columns')
        self.matrix = matrix

    def dimensions(self):
        return self.rows, self.cols

    def transpose(self):
        matrix_t = []
        for col_index in range(self.cols):
            row_t = [self.matrix[row_index][col_index] for row_index in range(self.rows)]
            matrix_t.append(row_t)
        return Matrix(matrix_t)

    def __mul__(self, vector):
        if self.cols != len(vector):
            raise Exception('Incompatible matrix/vector dimensions')
        return Vector([list_sum(list_mul(row, vector.vector)) for row in self.matrix])

    def print(self):
        for row in self.matrix:
            print(row)


if __name__ == '__main__':
    v = Vector([1, 2, 3])
    print(len(v))
    v.print()
    print(v * v)
    (v + v).print()

    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(m.dimensions())
    m.print()
    m.transpose().print()

    m = Matrix([[1, 1], [1, 2]])
    v = Vector([3, 4])
    (m * v).print()
