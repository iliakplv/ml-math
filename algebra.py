def list_sum(l):
    result = 0
    for item in l:
        result += item
    return result


def list_mul(l1, l2):
    return [l1[i] * l2[i] for i in range(len(l1))]


def mat_mul(matrix1, matrix2):
    matrix1rows = len(matrix1)
    matrix1cols = len(matrix1[0])
    matrix2cols = len(matrix2[0])

    result = []
    for row in range(matrix1rows):
        result_line = []
        for col in range(matrix2cols):
            cross_sum = 0
            for sum_index in range(matrix1cols):
                cross_sum += matrix1[row][sum_index] * matrix2[sum_index][col]
            result_line.append(cross_sum)
        result.append(result_line)

    return result


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

    def __sub__(self, other):
        if len(self) != len(other):
            raise Exception('Different size vectors')
        return Vector([self.vector[i] - other.vector[i] for i in range(len(self))])

    def __mul__(self, other):
        """
        Dot product OR multiplication by scalar
        :param other: Vector OR float
        :return: scalar value OR Vector (respectively)
        """
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise Exception('Different size vectors')
            return list_sum(list_mul(self.vector, other.vector))
        else:
            return self.mul_scalar(other)

    def mul_element_wise(self, other):
        """
        Element-wise product
        :param other: Vector
        :return: Vector
        """
        if len(self) != len(other):
            raise Exception('Different size vectors')
        return Vector(list_mul(self.vector, other.vector))

    def mul_outer(self, other):
        """
        Outer product
        :param other: Vector
        :return: Matrix
        """
        m1 = Matrix([[item] for item in self.vector])
        m2 = Matrix([other.vector])
        return m1 * m2

    def mul_scalar(self, value):
        return Vector([item * value for item in self.vector])

    def __truediv__(self, value):
        """
        Divide by scalar value
        :param value: scalar
        :return: Vector
        """
        if value == 0:
            raise Exception('Division by zero')
        return Vector([item / value for item in self.vector])

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

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise Exception('Matrix dimension don\'t match')
        return Matrix([[self.matrix[i][j] - other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, other):
        """
        Multiply by vector OR matrix OR scalar
        :param other: Vector OR Matrix OR float
        :return: Vector OR Matrix OR Matrix (respectively)
        """
        if isinstance(other, Vector):
            return self.mul_vector(other)
        elif isinstance(other, Matrix):
            return self.mul_matrix(other)
        else:
            return self.mul_scalar(other)

    def mul_vector(self, vector):
        if self.cols != len(vector):
            raise Exception('Incompatible matrix/vector dimensions')
        return Vector([list_sum(list_mul(row, vector.vector)) for row in self.matrix])

    def mul_matrix(self, other):
        if self.cols != other.rows:
            raise Exception('Incompatible matrix dimensions')
        return Matrix(mat_mul(self.matrix, other.matrix))

    def mul_scalar(self, value):
        return Matrix([[item * value for item in row] for row in self.matrix])

    def __truediv__(self, value):
        """
        Divide by scalar value
        :param value: scalar
        :return: Matrix
        """
        if value == 0:
            raise Exception('Division by zero')
        return Matrix([[item / value for item in row] for row in self.matrix])

    def print(self):
        for row in self.matrix:
            print(row)


if __name__ == '__main__':
    v = Vector([1, 2, 3])
    v.print()
    (v / 10).print()
    v.mul_scalar(10).print()
    print(len(v))
    (v + v).print()
    (v - v).print()
    print(v * v)
    v.mul_element_wise(v).print()
    v.mul_outer(v).print()
    (v.mul_outer(v) / 10).print()

    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(m.dimensions())
    m.print()
    m.transpose().print()

    m = Matrix([[1, 1], [1, 2]])
    (m * 10).print()
    ((m * 10) - m).print()
    v = Vector([3, 4])
    (m * v).print()
    (m * m).print()
