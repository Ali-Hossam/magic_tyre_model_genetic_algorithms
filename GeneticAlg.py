import numpy as np
import os

class GeneticAlgorithm:
    def __init__(self, NP, CP, MP, F, mode, slip_arr, Y_mes, Fz, n_params, itermax, c):
        """Genetic algorithm to estimate the parameters of Pacejka Magic tire 
        model in both longitudinal and lateral pure slip condition, also combined
        slip conditions can be added later

            Args:
                NP (int): Number of population
                CP (float): Cross over probability
                MP (float): Mutation probability
                F (float): Real value that controls the disturbance of the best individual
                mode (str): lat or long, for lateral and longitudinal pure slip mode
                slip_arr (numpy array): array of longitudinal slip, or side slip values
                Y_mes (numpy array): array of the measured forces
                Fz (float): Vertical force on the tire
                n_params (int): number of parameters to be estimated, which is 
                14 in longitudinal forces and 18 in lateral forces.
                
                itermax (int): max number of iteration before increasing the threshold
                c (float): cumber angle
        """

        self.NP = NP    # number of population
        self.CP = CP    # Cross over probability
        self.MP = MP    # Mutation probability
        self.F = F  # real value that controls the disturbance of the best individual
        self.itermax = itermax  # number of iterations of the algorithm

        # numpy array of errors for every individual
        self.errors = np.zeros(NP)
        self.population = 0  # numpy array of population
        self.V = 0  # Vector created in the selection step
        self.x_best = 0  # array of the best parameters

        self.n_params = n_params  # number of parameters
        # mode (string): modes of the fitness_func can be pure or combined
        # lateral and longitudinal forces.
        self.mode = mode

        # array of longitudinal slip or side slip angles.
        self.slip_arr = slip_arr

        # array of true measured forces for every slip angle
        self.Y_mes = Y_mes

        # Vertical force on the tire, which is constant as we assumed that the
        # acceleration is constant
        self.Fz = Fz

        # data is a list to store sum squared error every iteration for evaluation purpose
        self.data = []

        # c: camber angle which is used in lateral forces
        self.c = c

        # min_error: stores minimum error
        self.min_error = np.inf

    def init_pop(self):
        """initializes the starting population

            Args:
                n_params (int): number of parameters
        """
        n_params = self.n_params

        # generate random values between zero and one for the population
        self.population = np.random.rand(self.NP, n_params)

    def error_metric(self, true_value, pred_value):
        """absolute percentage error"""
        return np.abs((true_value - pred_value)/true_value) * 100

    def calc_fitness(self, individual):
        """calculates the fitness error of some individual"""
        mode = self.mode  # [pure long, pure lat./, combined long, combined lat.]
        Fz = self.Fz    # vertical force
        Y_mes = self.Y_mes  # measured force
        error = 0   # fitness error

        if mode == "Long":   # Pure Longitudinal Slip
            b0, b1, b2, b3, b4, b5, b6,\
                b7, b8, b9, b10, b11, b12, b13 = individual

            for j, slip_angle in enumerate(self.slip_arr):
                D = b1 * pow(Fz, 2) + b2 * Fz
                C = b0
                B = ((b3 * pow(Fz, 2) + b4 * Fz) * np.exp(-b5*Fz))/(C*D)
                Shx = b9 * Fz + b10
                Svx = b11 * Fz + b12
                E = (b6 * pow(Fz, 2) + b7*Fz + b8) * \
                    (1 - b13 * np.sign(slip_angle + Shx))

                x = slip_angle
                y = y = D * np.sin(C * np.arctan(B * x - E *
                                   (B*x - np.arctan(B*x)))) + Svx

                # Longitudinal force in pure slip condition
                Y_pure = y / (x + Shx)
                error += self.error_metric(Y_mes[j], Y_pure)

        elif mode == "Lat":
            a0, a1, a2, a3, a4, a5, a6, \
                a7, a8, a9, a10, a11, a12,\
                a13, a14, a15, a16, a17 = individual

            c = self.c
            for j, slip_angle in enumerate(self.slip_arr):
                D = Fz*(a1*Fz + a2) * (1 - a3 * pow(c, 2))
                C = a0
                B = (a4 * np.sin(2*np.arctan(Fz/a5) * (1 - a6*np.abs(c))))/(C*D)
                Shy = a11*Fz + a12 + a13*c
                Svy = a14*Fz + a15 + c*(a16*pow(Fz, 2) + a17 * Fz)
                E = (a7 * Fz + a8)*(1 - (a9*c + a10 * np.sign(slip_angle + Shy)))

                x = slip_angle
                y = D * np.sin(C * np.arctan(B * x - E *
                                             (B * x - np.arctan(B*x)))) + Svy

                # Longitudinal force in pure slip condition
                Y_pure = y / (x + Shy)
                error += self.error_metric(Y_mes[j], Y_pure)

        return error/len(self.slip_arr)

    def calc_goal_func(self):
        """calculates error for every individual of the population based on some
        function.
        """
        # reset population errors
        self.errors = np.zeros(self.NP)
        for i, individual in enumerate(self.population):
            error = self.calc_fitness(individual)

            # update individual error
            self.errors[i] = error

    def selection(self):
        """selects two random individuals and the best individual and they make 
        up a disturbing vector V."""
        x1, x2 = np.random.randint(
            0, self.NP, size=2)  # random individuals indices

        # random individuals
        x1, x2 = self.population[x1], self.population[x2]
        # best individual of the population
        self.x_best = self.population[np.argmin(self.errors)]
        self.min_error = min(self.errors)
        self.data.append(min(self.errors))
        self.V = self.x_best + self.F * (x1 - x2)

    def crossover(self, individual, prev_error):
        """reproduction of individual descendants

            Args:
                prev_error(float): previous error of the passed individual   
        """
        r = np.random.random()
        if r < self.CP:
            # init child
            child = np.zeros(self.n_params)

            # generate random indices
            indices = np.random.randint(
                0, self.n_params, size=np.random.randint(1, self.n_params-1))

            # create the child
            for i in indices:
                child[i] = individual[i]

            zeros = np.where(child == 0)[0]
            for i in zeros:
                child[i] = self.V[i]

            # calculate new child's error
            error = self.calc_fitness(child)
            if error < prev_error:
                return child
            else:
                return individual

        return individual

    def mutate(self, individual):
        """perform mutation which is a random change of gene during reproduction"""
        r = np.random.random()
        if r < self.MP:
            # improvised
            for i in range(2):
                m = np.random.randint(0, self.n_params)
                individual[m] += np.random.uniform(-1, 1)
        return individual

    def crossover_mutation(self):
        """apply crossover and mutation to the population"""
        new_population = np.zeros(
            self.population.shape)    # array to store new population

        for idx, individual in enumerate(self.population):
            error = self.errors[idx]
            new_population[idx] = self.crossover(individual, error)
            new_population[idx] = self.mutate(new_population[idx])

        # update population
        self.population = new_population

    def apply_alg(self):
        """applies genetic algorithm in one function"""
        self.init_pop()

        min_error = 2
        iterr = 0
        while self.min_error > min_error:
            # after 500 iterations if the error is not accepted increase the threshold
            iterr += 1
            if iterr == self.itermax:
                self.init_pop()
                iterr = 0
                min_error += 1

            self.calc_goal_func()
            self.selection()
            self.crossover_mutation()
            # print(self.min_error)

        # update x_best
        self.selection()


if __name__ == '__main__':
    # Get the current working directory
    current_directory = os.path.dirname(__file__)
    results_dir = current_directory + '\\results\\'
    print(results_dir)
    
    # import data
    DIR_PATH  = os.path.dirname(os.path.abspath(__file__))
    Fz = np.loadtxt(os.path.join(DIR_PATH,'test_data\Fz.txt'), dtype=float)
    slipAngle = np.loadtxt(os.path.join(DIR_PATH,'test_data\slipAngle.txt'), dtype=float)
    Y_mes = np.loadtxt(os.path.join(DIR_PATH,'test_data\Y_mes.txt'), dtype=float)

    # define constants
    mode = "Long"
    NP = 100
    CP = 0.6
    MP = 0.1
    F = 0.4
    n_params = 14
    iter_max = 1000
    c = 4
    
    # Algorithm
    alg = GeneticAlgorithm(100, 0.6, 0.1, 0.4, mode,
                           slipAngle, Y_mes, Fz, 14, 1000, c)
    alg.apply_alg()

    # save parameters and error data
    np.savetxt(results_dir + 'params.txt', alg.x_best, fmt='%.8f')
    np.savetxt(results_dir + 'errors.txt', alg.data, fmt='%.8f')

# best params for longitudinal
# # alg params
# mode = "Long"
# NP = 50
# CP = 0.6
# MP = 0.1
# F = 0.5
# n_params = 14
# iter_max = 1000
# c = 0

# or
# alg params
# mode = "Long"
# NP = 100
# CP = 0.6
# MP = 0.3
# F = 0.4
# n_params = 14
# iter_max = 1000
# c = 0
