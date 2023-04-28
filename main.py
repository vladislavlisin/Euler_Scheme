import numpy as np
import scipy as sp
import random

def func_f(x, t, a=0.027):
    return x + 2*t - np.exp(x) + 12*a*(x**2) + a*t*np.exp(x)


def left_cond_t(t):
    return t ** 2 - t


def right_cond_t(t):
    return t ** 2 + t - t * np.exp(1)


def cond_x(x):
    return -x ** 4 + x


def func_u(x, t):
    return -x ** 4 + x + t * x + t ** 2 - t * np.exp(x)


# u^k_j = u(t^k,x_j)
# j = 1, ... , N
# k = 1, ... , M
def Euler_scheme(N, delta_t, M, delta_x, ak, bk, ck):
    u_values = np.empty([1])
    u_values = np.delete(u_values, 0)
    # create array of psi_0_k with t = 0 and x with j = non-const, 1 and 0 are not included
    psi_j_k = np.array([cond_x(j) + delta_t * func_f(j, delta_t) for j in np.arange(delta_x, 1, delta_x)])
    print(psi_j_k)
    # fix k t_value
    for k in np.arange(0, 1+delta_t, delta_t):
        if k > 1:
            continue
        alpha_1 = alpha_prev = 0  # init first alpha
        betta_1 = betta_prev = left_cond_t(k+delta_t)  # init first betta
        alpha_arr = np.array([alpha_1])
        betta_arr = np.array([betta_1])

        # find u_values in x_value (j; 0,1,2,3, ...)
        for j in range(0, N-2):  # remember N-1 actually (start and end are not included)
            alpha_k = -(ak / (bk + ck * alpha_prev))  # calc alpha_k
            betta_k = ((psi_j_k[j] - ck * betta_prev) / (bk + ck * alpha_prev))  # calc betta_k

            alpha_arr = np.append(alpha_arr, alpha_k)
            betta_arr = np.append(betta_arr, betta_k)

            alpha_prev = alpha_k  # update alpha
            betta_prev = betta_k  # update betta

        u_last = right_cond_t(k+delta_t)  # use start right cond to calc last value of u
        u_values_step = np.array([u_last])
        psi_j_k = 0  # zeroes that vector
        psi_j_k = np.delete(psi_j_k, 0)

        for j in range(N-2, 0, -1):  # array boards: (0, 11)
            u_value = alpha_arr[j] * u_last + betta_arr[j]  # calc u value with (k,j+1)
            u_values_step = np.append(u_values_step, u_value)  # add this value to grid array
            u_last = u_value
            psi_j_k = np.append(psi_j_k, u_last + delta_t * func_f(j*delta_x, k+delta_t))  # in point (j+1, k) (x, t) update

        u_values_step = np.append(u_values_step, left_cond_t(k))
        # reverse
        psi_j_k = np.flip(psi_j_k, 0)
        u_values = np.append(u_values, np.flip(u_values_step, 0))
        # print(u_values_step)

    return u_values


# u^k_j = u(x_j, t^k)
# j = 1, ... , N
# k = 1, ... , M
# for us is M = N
def Euler_scheme_v_2(N, delta_t, M, delta_x, ak, bk, ck):
    # calc all start conditions
    steps_t = [t for t in np.arange(0, 1 + 2*delta_t, delta_t) if t <= 1]
    steps_x = [h for h in np.arange(0, 1 + 2*delta_x, delta_x) if h <= 1]
    u_x_0 = [cond_x(k) for k in steps_x]
    u_0_t = [left_cond_t(k) for k in steps_t]
    u_1_t = [right_cond_t(k) for k in steps_t]


    # create an empty array for u values
    u_values = np.array(u_x_0)

    # create array of psi_0_k with t = 0 and x with j = non-const, EDGES ARE NOT INCLUDED
    psi_j_k = np.array([u_x_0[j] + delta_t * func_f(steps_x[j], delta_t) for j in range(0, M-2)])

    # print(0)
    # print(u_values)

    k = 0  # n = 0
    # fix t and go through x_j
    # indices from 0 to N
    while(k < N-1):
        # print(k + 1)
        alpha_1 = 0
        betta_1 = u_0_t[k+1]
        alpha_arr = np.array([alpha_1])
        betta_arr = np.array([betta_1])

        #  move from left to right, calc alphas and bettas
        j = 0  # skip first value 0
        while(j < N-2):
            alpha_k = -(ak / (bk + ck * alpha_arr[j-1]))  # calc alpha_k
            betta_k = ((psi_j_k[j] - ck * betta_arr[j-1]) / (bk + ck * alpha_arr[j-1]))  # calc betta_k

            alpha_arr = np.append(alpha_arr, alpha_k)
            betta_arr = np.append(betta_arr, betta_k)
            j += 1

        u_values_step = np.array([u_1_t[k+1]])
        u_last = u_1_t[k+1]

        psi_j_k = 0  # zeroes that vector
        psi_j_k = np.delete(psi_j_k, 0)

        i = N - 2
        while(i >= 0):
            u_value = alpha_arr[i]*u_last + betta_arr[i]
            u_values_step = np.append(u_values_step, u_value)  # add this value to grid array
            u_last = u_value
            psi_j_k = np.append(psi_j_k,
                                u_last + delta_t*func_f(steps_x[i], steps_t[i+1]))  # in point (j+1, k) (x, t) update
            i -= 1

        # reverse
        psi_j_k = np.flip(psi_j_k, 0)  # need to reverse this vector
        u_values = np.append(u_values, np.flip(u_values_step, 0))
        # print(np.flip(u_values_step, 0))

        k += 1

    return u_values

# init all patterns for progonka

one = input()
two = input()

np.set_printoptions(precision=4, suppress=True)
a = 0.027
x0 = 0
xj = 1
t0 = 0
tk = 1
print("Enter t nodes and x nodes numbers")
t_num = int(input())  # N
x_num = int(input())  # M
delta_t = tk / int(t_num + 1)  # step t
delta_x = xj / int(x_num + 1)  # step x


ak = (-(delta_t * a) / delta_x ** 2)
bk = (1 + (2 * a * delta_x) / delta_x ** 2)
ck = (-(delta_t * a) / delta_x ** 2)

print(delta_t, delta_x)
print(ak, bk, ck)

# points = Euler_scheme_v_2(t_num+2, delta_t, x_num+2, delta_x, ak, bk, ck)

# points = np.reshape(points, (x_num+2, t_num+2))
# print(points)
# print()
# points = points.T
# print(points)

# value in x = 0.5 and t = 0.1*k, k=0,1,2,3,4,...
# tru_u_value = np.array([func_u(0.5, t) for t in np.arange(0, 1+delta_t, delta_t) if t <= 1])
# cals_u_value = np.array([t for t in points[5]])
# print("tru u value ", tru_u_value)
# print("calc u value ", cals_u_value)





# stop stop stop

take = input()

points = Euler_scheme(t_num+2, delta_t, x_num+2, delta_x, ak, bk, ck)
take2 = input()
points = np.reshape(points, (x_num+2, t_num+2))
print(points)
print()
points = points.T
print(points)

points1 = points2 = points3 = points4 = np.array([0])

# value in x = 0.5 and t = 0.1*k, k=0,1,2,3,4,...
tru_u_value = np.array([func_u(0.5, t) for t in np.arange(0, 1+delta_t, delta_t) if t <= 1])
cals_u_value = np.array([t for t in points[5]])
diff = tru_u_value - cals_u_value
diff = abs(diff)
print(diff)

print("tru u value ", tru_u_value)
print("calc u value ", cals_u_value)


points = Euler_scheme(t_num+2, delta_t, x_num+2, delta_x, ak, bk, ck)
tru_u_value = np.array([func_u(0.5, t) for t in np.arange(0, 1+delta_t, delta_t) if t <= 1])
diff = tru_u_value - cals_u_value


a1 = max(diff)
a2 = max(points2)
a3 = max(points3)
a4 = max(points4)

take3 = input()
print(a1, a2, a3, a4)


