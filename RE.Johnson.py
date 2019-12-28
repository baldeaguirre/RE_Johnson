## Python script based in RE.Johnson.R function from R-project
import math
import numpy as np
from random import seed
from scipy.stats import norm, anderson

## VECTORES DE PRUEBA ##
seed(1)
#x = np.random.uniform(-1, 0, 100)
#x = np.random.gamma(2, 1, 100)
#x = np.random.weibull(5, 100)
x = [1.549344, 1.546645, 1.544776, 1.543909, 1.543455, 1.545196, 1.54384, 1.543898, 1.543805, 1.544422, 1.543802, 1.543654, 1.543546, 1.543814, 1.543343, 1.543367, 1.543279, 1.543877, 1.544108, 1.544108, 1.543921, 1.543722, 1.543355, 1.54417, 1.54332, 1.543202, 1.54318, 1.544055, 1.544168, 1.543752, 1.542831, 1.542336, 1.543276, 1.543337, 1.543535, 1.542988, 1.543434, 1.543373, 1.544115, 1.543607, 1.543175, 1.544434]
#x = [0.874974924, 0.739689286, 0.126436953, 0.109543059, 0.522303574, 0.474649526, 0.613349210, 0.339495443, 0.379904679, 0.466630894, 0.787707971, 0.377409512, 0.136373097, 0.600473308, 0.843346465, 0.330599963, 0.304702462, 0.728782165, 0.968097989, 0.857731216, 0.894255504, 0.201960752, 0.327181479, 0.070106104, 0.529454106, 0.567801285, 0.823306670, 0.219090167, 0.762969622, 0.881070242, 0.752766070, 0.314637363, 0.835653048, 0.931615065, 0.001220706, 0.837766316, 0.561240475, 0.569746322, 0.612696547, 0.100529598, 0.761149222, 0.461309014, 0.657934088, 0.898015163, 0.361464564, 0.974976033, 0.812739759, 0.238852778, 0.813335280, 0.564138781] # Distribuciṕn Uniforme
#x = [1.2236462, 0.2349167, 2.2498103, 0.5718354, 2.0477937, 1.0482141, 2.2648465, 3.8622847, 2.2676600, 0.2522457, 1.2757665, 5.8821105, 4.3359184, 1.9880159, 0.3552512, 1.6772685, 1.6257625, 3.9105471, 0.8398705, 3.2814825, 0.9523483, 0.2202508, 2.4667064, 3.3721659, 3.1268380, 1.8159036, 2.8271892, 1.2385267, 1.8258532, 1.7932323, 0.5370760, 0.9169933, 2.1901186, 2.6684173, 1.2198107, 3.2041770, 2.3526026, 0.3063804, 0.9300882, 2.5530754, 3.3268206, 0.7947305, 2.1779574, 0.8313134, 1.1091836, 1.3458134, 1.2237923, 4.2222367, 1.3891295, 0.3065765] # Distribución Gamma
#x = [0.8017174, 0.8195303, 0.8656629, 0.9190050, 0.7585261, 0.7609965, 1.1877263, 0.8680847, 0.8224036, 0.8101624, 1.1081195, 1.2072886, 0.8973867, 0.7826767, 0.8769408, 1.3089377, 0.9501279, 0.9094235, 1.1438780, 0.7852159, 0.9507362, 1.1395710, 0.7972585, 1.0821259, 0.8075633, 1.0932120, 0.6612767, 0.8226804, 1.1584222, 0.8379464, 0.3915111, 0.8498298, 1.1979511, 0.9780915, 0.9888716, 0.7189232, 0.9035990, 0.9123447, 0.9695277, 0.9670831, 1.1225053, 1.0988788, 1.1208255, 1.4240213, 0.9322003, 0.7300996, 0.6515770, 0.8068353, 0.6477033, 0.5404665] # Distribución Weibull

def p_value_AD(x):
    AD = anderson(x)[0]
    n = len(x)
    AD = AD * (1 + (0.75 / n) + 2.25 / (n ** 2))  # Estadístico Anderson Darling ajustado
    if AD >= 0.6:
        p = math.exp(1.2937 - 5.709 * AD + 0.0186 * (AD ** 2))
    elif AD >= 0.34:
        p = math.exp(0.9177 - 4.279 * AD - 1.38 * (AD ** 2))
    elif AD > 0.2:
        p = 1 - math.exp(-8.318 + 42.796 * AD - 59.938 * (AD ** 2))
    else:
        p = 1 - math.exp(-13.436 + 101.14 * AD - 223.73 * (AD ** 2))
    return p

def Johnson(x):
    sort_x = sorted(x)
    #######VARIABLES######
    z = np.arange(0.25, 1.26, 0.01)  # z values
    QR = np.zeros((1, 101))
    q = np.zeros((101, 4)) # quartile
    j = np.zeros((101, 4)) # element of x relative to q
    y = np.zeros((101, 4))

    xl = np.zeros((101, 1))
    xm = np.zeros((101, 1))
    xu = np.zeros((101, 1))

    b_eta = np.zeros((101, 1))
    b_gamma = np.zeros((101, 1))
    b_lambda = np.zeros((101, 1))
    b_epsilon = np.zeros((101, 1))

    l_eta = np.zeros((101, 1))
    l_gamma = np.zeros((101, 1))
    l_lambda = np.zeros((101, 1))
    l_epsilon = np.zeros((101, 1))

    u_eta = np.zeros((101, 1))
    u_gamma = np.zeros((101, 1))
    u_lambda = np.zeros((101, 1))
    u_epsilon = np.zeros((101, 1))

    xsb = np.zeros((len(x), 101))
    xsl = np.zeros((len(x), 101))
    xsu = np.zeros((len(x), 101))

    xsb_valida = np.zeros((1, 101))
    xsl_valida = np.zeros((1, 101))
    xsu_valida = np.zeros((1, 101))

    xsb_adtest = np.zeros((1, 101))
    xsl_adtest = np.zeros((1, 101))
    xsu_adtest = np.zeros((1, 101))

    f_gamma = 0
    f_lambda = 0
    f_epsilon = 0
    f_eta = 0
    #################################

    for i in range(101):
        q[i, 0] = norm.cdf(-3 * z[i])
        q[i, 1] = norm.cdf(-1 * z[i])
        q[i, 2] = norm.cdf(1 * z[i])
        q[i, 3] = norm.cdf(3 * z[i])

        j[i, 0] = len(x) * q[i, 0] + 0.5
        j[i, 1] = len(x) * q[i, 1] + 0.5
        j[i, 2] = len(x) * q[i, 2] + 0.5
        j[i, 3] = len(x) * q[i, 3] + 0.5

        if(j[i, 0] < 1):
            y[i, 0] = min(sort_x)
        else:
            y[i, 0] = (sort_x[math.ceil(j[i, 0]) -  1] - sort_x[math.floor(j[i, 0]) - 1]) / (math.ceil(j[i, 0]) - math.floor(j[i, 0])) * (j[i, 0] - math.floor(j[i, 0])) + sort_x[math.floor(j[i, 0] - 1)]
        if(j[i, 1] > len(x)):
            y[i, 1] = max(sort_x)
        else:
            y[i, 1] = (sort_x[math.ceil(j[i, 1]) - 1] - sort_x[math.floor(j[i, 1]) - 1]) / (math.ceil(j[i, 1]) - math.floor(j[i, 1])) * (j[i, 1] - math.floor(j[i, 1])) + sort_x[math.floor(j[i, 1] - 1)]
        if(j[i, 2] > len(x)):
            y[i, 2] = max(sort_x)
        else:
            y[i, 2] = (sort_x[math.ceil(j[i, 2]) - 1] - sort_x[math.floor(j[i, 2]) - 1]) / (math.ceil(j[i, 2]) - math.floor(j[i, 2])) * (j[i, 2] - math.floor(j[i, 2])) + sort_x[math.floor(j[i, 2] - 1)]
        if(j[i, 3] > len(x)):
            y[i, 3] = max(sort_x)
        else:
            y[i, 3] = (sort_x[math.ceil(j[i, 3]) - 1] - sort_x[math.floor(j[i, 3]) - 1]) / (math.ceil(j[i, 3]) - math.floor(j[i, 3])) * (j[i, 3] - math.floor(j[i, 3])) + sort_x[math.floor(j[i, 3] - 1)]

        QR[0][i] = ((y[i, 3] - y[i, 2]) * (y[i, 1] - y[i, 0])) / ((y[i, 2] - y[i, 1]) ** 2)

        xl[i, 0] = y[i, 1] - y[i, 0]
        xm[i, 0] = y[i, 2] - y[i, 1]
        xu[i, 0] = y[i, 3] - y[i, 2]

    ######### SB,SL,SU
    for i in range(101):
        if(0.5 * (((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i])) ** 0.5) < 1):
            b_eta[i, 0] = -1000
        else:
            b_eta[i, 0] = z[i] / (math.acosh(0.5 * (((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i])) ** 0.5)))
        if(0.5 * (((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i])) ** 0.5) < 1):
            b_gamma[i, 0] = -1000
        else:
            b_gamma[i, 0] = b_eta[i, 0] * np.arcsinh(((xm[i] / xl[i] - xm[i] / xu[i]) * (((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i]) - 4) ** 0.5)) / (2 * (((xm[i] ** 2) / (xl[i] * xu[i])) - 1)))
        if(((((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i]) - 2) ** 2) - 4) < 0):
            b_lambda[i, 0] = 1000
        else:
            b_lambda[i, 0] = (xm[i] * (((((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i]) - 2) ** 2) - 4) ** 0.5)) / (((xm[i] ** 2) / (xl[i] * xu[i])) - 1)
        if(((((1 + xm[i] / xu[i]) * (1 + xm[i] / xl[i]) - 2) ** 2) - 4) < 0):
            b_epsilon[i, 0] = 1000
        else:
            b_epsilon[i, 0] = 0.5 * (y[i, 1] + y[i, 2] - b_lambda[i, 0] + ((xm[i] * (xm[i] / xl[i] - xm[i] / xu[i])) / (((xm[i] ** 2) / (xl[i] * xu[i])) - 1)))

        l_eta[i] = 2 * z[i] / (np.log(xu[i] / xm[i]))
        if((xu[i] / xm[i] - 1) / ((xu[i] * xm[i]) ** 0.5) <= 0):
            l_gamma[i] = 1000
        else:
            l_gamma[i] = l_eta[i] * np.log((xu[i] / xm[i] - 1) / ((xu[i] * xm[i]) ** 0.5))
        l_epsilon[i] = 0.5 * (y[i, 1] + y[i, 2] - xm[i] * ((xu[i] / xm[i] + 1) / (xu[i] / xm[i] - 1)))

        if((0.5 * (xu[i] / xm[i] + xl[i] / xm[i])) < 1):
            u_eta[i, 0] = -1000
        else:
            u_eta[i, 0] = 2 * z[i] / (np.arccosh(.5 * (xu[i] / xm[i] + xl[i] / xm[i])))
        if((xu[i] * xl[i] / (xm[i] ** 2)) - 1 < 0):
            u_gamma[i, 0] = -1000
        else:
            u_gamma[i, 0] = u_eta[i] * np.arcsinh((xl[i] / xm[i] - xu[i] / xm[i]) / (2 * (((xu[i] * xl[i] / (xm[i] ** 2)) - 1) ** 0.5)))
        if((xu[i] * xl[i] / (xm[i] ** 2)) - 1 < 0):
            u_lambda[i, 0] = 1000
        else:
            u_lambda[i] = (2 * xm[i] * (((xu[i] * xl[i] / (xm[i] ** 2)) - 1) ** 0.5)) / ((xu[i] / xm[i] + xl[i] / xm[i] - 2) * ((xu[i] / xm[i] + xl[i] / xm[i] + 2) ** 0.5))
        u_epsilon[i] = 0.5 * (y[i, 1] + y[i, 2] + ((xm[i] * (xl[i] / xm[i] - xu[i] / xm[i])) / ((xu[i] / xm[i] + xl[i] / xm[i]) - 2)))

        for o in range(len(x)):
            if((x[o] - b_epsilon[i]) / (b_lambda[i] + b_epsilon[i] - x[o]) <= 0):
                xsb_valida[0, i] = xsb_valida[0, i] + 1
            else:
                xsb[o, i] = b_gamma[i] + b_eta[i] * np.log((x[o] - b_epsilon[i]) / (b_lambda[i] + b_epsilon[i] - x[o]))
            if((x[o] - l_epsilon[i]) <= 0):
                xsl_valida[0, i] = xsl_valida[0, i] + 1
            else:
                xsl[o, i] = l_gamma[i] + l_eta[i] * np.log(x[o] - l_epsilon[i])
            if((xu[i] * xl[i] / (xm[i] ** 2) - 1) < 0):
                xsu_valida[0, i] = xsu_valida[0, i] + 1
            else:
                xsu[o, i] = u_gamma[i] + u_eta[i] * np.arcsinh((x[o] - u_epsilon[i]) / u_lambda[i])

        if (xsb_valida[0, i] == 0): xsb_adtest[0, i] = p_value_AD(xsb[:, i])
        if (xsl_valida[0, i] == 0): xsl_adtest[0, i] = p_value_AD(xsl[:, i])
        if (xsu_valida[0, i] == 0): xsu_adtest[0, i] = p_value_AD(xsu[:, i])

    # Se convierten valores 'NaN' a cero
    xsb_adtest = np.nan_to_num(xsb_adtest)
    xsl_adtest = np.nan_to_num(xsl_adtest)
    xsu_adtest = np.nan_to_num(xsu_adtest)

    if(np.amax(xsb_adtest) > np.amax(xsl_adtest) and np.amax(xsb_adtest) > np.amax(xsu_adtest)):
        #dic = dict(p=np.amax(xsb_adtest), fun="SB", transformed=xsb[:, np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])], f_gamma=b_gamma[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0], f_lambda=b_lambda[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0], f_epsilon=b_epsilon[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0], f_eta=b_eta[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0])
        dic = {
            'p': np.amax(xsb_adtest),
            'fun': "SB",
            'transformed': xsb[:, np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])],
            'f_gamma': b_gamma[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0],
            'f_lambda': b_lambda[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0],
            'f_epsilon': b_epsilon[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0],
            'f_eta': b_eta[np.random.choice(np.where(xsb_adtest == np.amax(xsb_adtest))[1])][0]
        }
    elif(np.amax(xsl_adtest) > np.amax(xsu_adtest)):
        #dic = dict(p=np.amax(xsl_adtest), fun="SL", transformed=xsl[:, np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])], f_gamma=l_gamma[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0], f_lambda=l_lambda[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0], f_epsilon=l_epsilon[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0], f_eta=l_eta[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0])
        dic = {
            'p': np.amax(xsl_adtest),
            'fun': "SL",
            'transformed': xsl[:, np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])],
            'f_gamma': l_gamma[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0],
            'f_lambda': l_lambda[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0],
            'f_epsilon': l_epsilon[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0],
            'f_eta': l_eta[np.random.choice(np.where(xsl_adtest == np.amax(xsl_adtest))[1])][0]
        }
    else:
        #dic = dict(p=np.amax(xsu_adtest), fun="SU", transformed=xsu[:, np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])], f_gamma=u_gamma[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0], f_lambda=u_lambda[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0], f_epsilon=u_epsilon[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0], f_eta=u_eta[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0])
        dic = {
            'p': np.amax(xsu_adtest),
            'fun': "SU",
            'transformed': xsu[:, np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])],
            'f_gamma': u_gamma[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0],
            'f_lambda': u_lambda[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0],
            'f_epsilon': u_epsilon[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0],
            'f_eta': u_eta[np.random.choice(np.where(xsu_adtest == np.amax(xsu_adtest))[1])][0]
        }

    outDir = dict(JohnsonTransformation='Johnson Transformation', function=dic['fun'], p=dic['p'], transformed=dic['transformed'], f_gamma=dic['f_gamma'], f_lambda=dic['f_lambda'], f_epsilon=dic['f_epsilon'], f_eta=dic['f_eta'])
    return outDir

print(Johnson(x)['JohnsonTransformation'], '\n')
print('function:', Johnson(x)['function'], '\n')
print('p-value:', Johnson(x)['p'], '\n')
print(Johnson(x)['transformed'], '\n')
print('f_gamma:', Johnson(x)['f_gamma'], '\n')
print('f_lambda:', Johnson(x)['f_lambda'], '\n')
print('f_epsilon:', Johnson(x)['f_epsilon'], '\n')
print('f_eta:', Johnson(x)['f_eta'])