"""
    PAT-PNRR Optimizer
    Particle Swarm Optimization of MPE targets
    Francesco Melchiori, 2024
"""


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from matplotlib import cbook
from matplotlib.colors import LightSource
from matplotlib import cm

# import cv2

from pat_pnrr_mpe.pat_pnrr_comuni_excel_mapping import *
from pat_pnrr_mpe import pat_pnrr_3a_misurazione as pat_pnrr_3a
from pat_pnrr_mpe import pat_pnrr_4a_misurazione as pat_pnrr_4a
from pat_pnrr_mpe import pat_pnrr_5a_misurazione as pat_pnrr_5a


class ParameterSampling:

    def __init__(self, lowerbound, upperbound, samples_amount=2,
                 sampling_type='linear'):
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.samples_amount = samples_amount
        # if samples_amount <= 255:
        #     self.samples_amount = samples_amount
        # else:
        #     self.samples_amount = 255
        self.sampling_type = sampling_type
        self.samples_type = np.int16
        # if type(float()) in (type(self.lowerbound), type(self.upperbound)):
        #     self.samples_type = np.float16
        # else:
        #     self.samples_type = np.int16
        self.samples = np.zeros(shape=self.samples_amount,
                                dtype=self.samples_type)
        if sampling_type == 'linear':
            self.linear_sampling()

    def __repr__(self):
        print_message = ''
        print_message += "'{0}'".format(self.samples)
        return print_message

    def __plot__(self):
        plt.plot(np.linspace(start=0, stop=self.samples_amount,
                             num=self.samples_amount, endpoint=False),
                 self.samples, color='black', linestyle='None', marker='o')
        plt.show()

    def linear_sampling(self):
        self.samples = np.linspace(start=self.lowerbound, stop=self.upperbound,
                                   num=self.samples_amount,
                                   dtype=self.samples_type)
        return self.samples


class Particle:

    def __init__(self, gain_function, parameters, solution_space_sizes,
                 inertial_weight=1., cognitive_weight=1., social_weight=1.,
                 serial_number=0):
        self.gain_function = gain_function
        self.parameters = parameters
        self.solution_sizes = solution_space_sizes
        self.weight = np.ones([3], dtype=np.float16)
        self.weight[0] = inertial_weight
        self.weight[1] = cognitive_weight
        self.weight[2] = social_weight
        self.serial_number = serial_number
        self.random = np.ones([3], dtype=np.float16)
        self.speed = self.init_position()
        self.position = self.init_position()
        self.best = self.init_position()
        self.best_value = False
        self.best_swarm = self.init_position()
        self.best_swarm_value = False
        self.samples = []

    def __repr__(self):
        print_message = ''
        if self.serial_number == 0:
            print_message += "    * Swarm\n"
            print_message += "        Best position: {0}\n" \
                             "".format(self.best_swarm)
            print_message += "        Best value: {0}" \
                             "".format(self.best_swarm_value)
        else:
            print_message += "    * Particle {0}\n" \
                             "".format(self.serial_number)
            print_message += "        Weights: {0}\n" \
                             "".format(self.weight)
            print_message += "        Randoms: {0}\n" \
                             "".format(self.random)
            print_message += "        Speed: {0}\n" \
                             "".format(self.speed)
            print_message += "        Position: {0}\n" \
                             "".format(self.position)
            print_message += "        Best position: {0}\n" \
                             "".format(self.best)
            print_message += "        Best value: {0}\n" \
                             "".format(self.best_value)
            print_message += "        Last sample: {0}" \
                             "".format(self.samples[-1])
        return print_message

    def set_random(self):
        self.random = np.ones([3], dtype=np.float16)
        self.random[1:] = np.random.random((1, 2))

    def init_position(self):
        return np.zeros([len(self.solution_sizes)], dtype=np.int16)

    def set_position(self, position):
        self.position = position

    def set_best_swarm(self, position):
        self.best_swarm = position

    def quantize_vector(self, vector, bumping=True):
        position_control = abs(vector) < self.solution_sizes
        position_correct_indexes = np.where(position_control == False)

        position_correct_edge_positive = vector[position_correct_indexes] > 0
        position_correct_edge_negative = vector[position_correct_indexes] < 0
        position_correct_edge_bumps = abs(vector[position_correct_indexes]) // \
            self.solution_sizes[position_correct_indexes]
        position_correct_edge_even_bumps = (position_correct_edge_bumps % 2) == 0
        position_correct_edge_distance = abs(vector[position_correct_indexes]) % \
            self.solution_sizes[position_correct_indexes]
        
        position_correct_zero_distance = position_correct_edge_positive == \
            position_correct_edge_even_bumps
        position_correct_size_distance = position_correct_edge_positive != \
            position_correct_edge_even_bumps

        vector_correction = np.zeros(vector[position_correct_indexes].shape[0])
        if bumping:
            # edge_positive | edge_even_bumps | position_correct
            #             1 |               1 |  zero + distance
            #             0 |               0 |  zero + distance
            #             1 |               0 |  size - distance
            #             0 |               1 |  size - distance
            vector_correction[position_correct_zero_distance] = \
                position_correct_edge_distance[position_correct_zero_distance]
            vector_correction[position_correct_size_distance] = \
                self.solution_sizes[position_correct_indexes][position_correct_size_distance] - \
                position_correct_edge_distance[position_correct_size_distance]
        else:
            vector_correction[position_correct_edge_negative] = \
                position_correct_edge_distance[position_correct_edge_negative]
            vector_correction[position_correct_edge_positive] = \
                self.solution_sizes[position_correct_indexes][position_correct_edge_positive] - \
                position_correct_edge_distance[position_correct_edge_positive]

        vector[position_correct_indexes] = vector_correction
        vector = np.floor(vector).astype(int)

        return vector

    def sample_gain_function(self):
        function_parameters = []
        sample_coordinate = []
        for dimension, coordinate in enumerate(self.position):
            function_parameters.append(
                self.parameters[dimension].samples[coordinate])
            sample_coordinate.append(coordinate)
        value = self.gain_function(function_parameters)
        sampled_values = [sample[-1] for sample in self.samples]
        if (self.best_value is False) or (value >= max(sampled_values)):
            self.best = self.position
            self.best_value = value
        sample = sample_coordinate
        sample.append(value)
        sample = tuple(sample)
        self.samples.append(sample)
        return sample

    def perturb(self):
        if self.best_value is False:
            intertial_displace = self.init_position()
            cognitive_displace = self.init_position()
            social_displace = self.init_position()
        else:
            intertial_displace = self.speed - 0
            cognitive_displace = self.best - self.position
            social_displace = self.best_swarm - self.position
        self.set_random()
        intertial_term = self.weight[0] * self.random[0] * intertial_displace
        cognitive_term = self.weight[1] * self.random[1] * cognitive_displace
        social_term = self.weight[2] * self.random[2] * social_displace
        self.speed = intertial_term + cognitive_term + social_term
        self.position = self.position + self.speed
        self.position = self.quantize_vector(self.position)
        sample = self.sample_gain_function()
        return sample


class PSO:

    def __init__(self, gain_function, parameters, iterations, particle_amount=3,
                 inertial_weight=1., cognitive_weight=1., social_weight=1.,
                 verbose=0):
        self.gain_function = gain_function
        self.parameters = parameters
        self.iterations = iterations
        self.particle_amount = particle_amount
        self.inertial_weight = inertial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.solution_sizes = np.array([parameter.samples.size
                                        for parameter in self.parameters],
                                       dtype=np.int16)
        self.particle_space = self.init_particle_space()
        self.particle_result = Particle(
            gain_function=self.gain_function, parameters=self.parameters,
            solution_space_sizes=self.solution_sizes,
            inertial_weight=self.inertial_weight,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight, serial_number=0)
        self.verbose = verbose

    def __repr__(self):
        print_message = ''
        solution_values = [self.parameters[i].samples[p]
                           for i, p
                           in enumerate(self.particle_result.best_swarm)]
        result_value = self.particle_result.best_swarm_value
        print_message += '    * Best result: {0}\n'.format(result_value)
        print_message += '    * Best solution: {0}'.format(solution_values)
        return print_message

    def init_particle_space(self):
        self.particle_space = [Particle(
            gain_function=self.gain_function, parameters=self.parameters,
            solution_space_sizes=self.solution_sizes,
            inertial_weight=self.inertial_weight,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight, serial_number=particle_number+1)
            for particle_number in range(self.particle_amount)]
        for particle in self.particle_space:
            position = np.array([random.randint(0, coordinate_size - 1)
                                 for coordinate_size
                                 in self.solution_sizes],
                                dtype=np.int16)
            particle.set_position(position)
        return self.particle_space

    def iter_particle_swarm(self):
        particles_data = []
        for particle in self.particle_space:
            particle_data = np.empty((len(self.solution_sizes) + 1,
                                      self.iterations))
            particles_data.append(particle_data)
        for i in range(self.iterations):
            if self.verbose >= 1:
                print('* Iteration {0}'.format(i+1))
            for particle, particle_data in zip(self.particle_space,
                                               particles_data):
                sample = particle.perturb()
                value = sample[-1]
                sampled_best_value = self.particle_result.best_swarm_value
                if value >= sampled_best_value:
                    self.particle_result.best_swarm = particle.position
                    self.particle_result.best_swarm_value = value
                particle_data[:, i] = sample
                if self.verbose >= 2:
                    print(particle)
            for particle in self.particle_space:
                particle.set_best_swarm(self.particle_result.best_swarm)
            if self.verbose >= 1:
                print(self)
        solution_values = [self.parameters[i].samples[p]
                           for i, p
                           in enumerate(self.particle_result.best_swarm)]
        return particles_data, solution_values
    

class MPE:
    def __init__(self, upperbound_comune=3, variation_comune=0.3):
        self.upperbound_comune = upperbound_comune
        self.variation_comune = variation_comune
        self.load_data()
        self.get_targets()
        self.serialize_params()

    def load_data(self):
        # LOAD DATA
        comuni_measures_dataframe_mpe_3 = pat_pnrr_3a.get_comuni_measures_dataframe(
            comuni_excel_map, load=True)
        comuni_measures_dataframe_mpe_4 = pat_pnrr_4a.get_comuni_measures_dataframe(
            comuni_excel_map, load=True)
        comuni_measures_dataframe_mpe_5 = pat_pnrr_5a.get_comuni_measures_dataframe(
            comuni_excel_map, load=True)

        # SELECT DATA and FILL NANs
        giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso = pd.concat([
            comuni_measures_dataframe_mpe_3[
                'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2022q3-4'],
            comuni_measures_dataframe_mpe_4[
                'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q1-2'],
            comuni_measures_dataframe_mpe_5[
                'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q3-4']],
            axis='columns', join='outer')
        giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.ffill(
            axis='columns', inplace=True)
        giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.bfill(
            axis='columns', inplace=True)
        giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.fillna(
            value=60, axis='columns', inplace=True)
        
        numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo = pd.concat([
            comuni_measures_dataframe_mpe_3[
                'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'],
            comuni_measures_dataframe_mpe_4[
                'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2'],
            comuni_measures_dataframe_mpe_5[
                'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4']],
            axis='columns', join='outer')
        numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo.ffill(
            axis='columns', inplace=True)
        
        giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso = pd.concat([
            comuni_measures_dataframe_mpe_3[
                'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'],
            comuni_measures_dataframe_mpe_4[
                'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'],
            comuni_measures_dataframe_mpe_5[
                'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q3-4']],
            axis='columns', join='outer')
        giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.ffill(
            axis='columns', inplace=True)
        giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.bfill(
            axis='columns', inplace=True)
        giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.fillna(
            value=60, axis='columns', inplace=True)
        
        numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo = pd.concat([
            comuni_measures_dataframe_mpe_3[
                'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4'],
            comuni_measures_dataframe_mpe_4[
                'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2'],
            comuni_measures_dataframe_mpe_5[
                'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q3-4']],
            axis='columns', join='outer')
        numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo.ffill(
            axis='columns', inplace=True)
        
        self.baseline_durata_pdc_ov = comuni_measures_dataframe_mpe_3[
            'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2022q3-4']\
            .mean()
        self.baseline_arretrato_pdc_ov = comuni_measures_dataframe_mpe_3[
            'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4']\
            .sum()
        self.baseline_durata_pds = comuni_measures_dataframe_mpe_3[
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4']\
            .mean()
        self.baseline_arretrato_pds = comuni_measures_dataframe_mpe_3[
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4']\
            .sum()
        
        self.comuni_measures_dataframe_mpe = pd.concat([
            giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.iloc[:, -1],
            numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo.iloc[:, -1],
            giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.iloc[:, -1],
            numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo.iloc[:, -1]],
            axis='columns', join='outer')
        
        (self.n_comuni, self.n_measures) = self.comuni_measures_dataframe_mpe.shape
        self.n_params = self.n_comuni * self.n_measures
        
        return self.comuni_measures_dataframe_mpe
    
    def get_targets(self):

        self.target_durata = -0.1
        self.target_arretrato = -0.15

        self.target_durata_pdc_ov = (self.baseline_durata_pdc_ov.round() * \
            (1 + self.target_durata)).round()
        self.target_arretrato_pdc_ov = (self.baseline_arretrato_pdc_ov.round() * \
            (1 + self.target_arretrato)).round()
        self.target_durata_pds = (self.baseline_durata_pds.round() * \
            (1 + self.target_durata)).round()
        self.target_arretrato_pds = (self.baseline_arretrato_pds.round() * \
            (1 + self.target_arretrato)).round()
        
        return True
    
    def serialize_params(self):
        """ serializzare, per tutti i comuni e per tutte le misure,
            le forchette di variazione di tutti questi parametri
            con ParameterSampling
        """

        params_lowerbound = np.floor(np.reshape(self.comuni_measures_dataframe_mpe.values,
            self.n_params) * (1 - self.variation_comune))
        params_upperbound = np.ceil(np.reshape(self.comuni_measures_dataframe_mpe.values,
            self.n_params) * (1 + self.variation_comune))
        params_upperbound[params_upperbound == 0] = self.upperbound_comune
        params_steps = (params_upperbound - params_lowerbound).astype(int) + 1

        self.params = [ParameterSampling(params_lowerbound[i_param],
                                         params_upperbound[i_param],
                                         params_steps[i_param])
                       for i_param in range(self.n_params)]

        return self.params
    
    def evaluate_solution(self, solution):
        
        solution_array = np.reshape(solution, (self.n_comuni, self.n_measures))
        solution_dataframe = pd.DataFrame(solution_array,
            index=self.comuni_measures_dataframe_mpe.index,
            columns=self.comuni_measures_dataframe_mpe.columns)
        
        solution_durata_pdc_ov = solution_dataframe.iloc[:, 0].mean()
        solution_arretrato_pdc_ov = solution_dataframe.iloc[:, 1].sum()
        solution_durata_pds = solution_dataframe.iloc[:, 2].mean()
        solution_arretrato_pds = solution_dataframe.iloc[:, 3].sum()

        print('Risultato:')
        print('    durata PdC OV: ' + str(np.ceil(solution_durata_pdc_ov).astype(int)))
        print('    arretrato PdC OV: ' + str(np.ceil(solution_arretrato_pdc_ov).astype(int)))
        print('    durata PdS: ' + str(np.ceil(solution_durata_pds).astype(int)))
        print('    arretrato PdS: ' + str(np.ceil(solution_arretrato_pds).astype(int)))
        
        return True

    def gaussian_gain_function(self, x, mu, sig):
            return ((1.0 / (sig * np.sqrt(2.0 * np.pi))) * \
                    np.exp(-np.power((x - mu), 2.0) / \
                            np.power((2.0 * sig), 2.0))
            )
    
    def gain_function(self, params):
        """ deserializzare la lista di parametri
            per associare ogni variazione ad ogni comune e misura
            per valutarne infine il valore della soluzione
            applicando tutte le variazioni
            calcolando il raggiungimento dei target
            al minor numero di comuni coinvolti (e
            ad almeno un comune per comunita' di valle e
            ad almeno un comune per cluster)
        """
        
        measure_changes_array = np.reshape(params, (self.n_comuni, self.n_measures))
        measure_changes_dataframe = pd.DataFrame(measure_changes_array,
            index=self.comuni_measures_dataframe_mpe.index,
            columns=self.comuni_measures_dataframe_mpe.columns)

        # dataframe mpe a cui applicare le variazioni
        comuni_changed_measures_dataframe = \
            self.comuni_measures_dataframe_mpe + \
            measure_changes_dataframe
        
        # dataframe da pavimentare fino a zero (no valori negativi)
        measure_zero_dataframe = pd.DataFrame(0,
            index=self.comuni_measures_dataframe_mpe.index,
            columns=self.comuni_measures_dataframe_mpe.columns)
        comuni_changed_measures_dataframe[
            comuni_changed_measures_dataframe < measure_zero_dataframe] = 0

        # misure target risultanti dalle variazioni
        self.changed_durata_pdc_ov = comuni_changed_measures_dataframe.iloc[:, 0].mean()
        self.changed_arretrato_pdc_ov = comuni_changed_measures_dataframe.iloc[:, 1].sum()
        self.changed_durata_pds = comuni_changed_measures_dataframe.iloc[:, 2].mean()
        self.changed_arretrato_pds = comuni_changed_measures_dataframe.iloc[:, 3].sum()

        # distanza complessiva tra le misure target risultanti ed i target finali
        gain = (self.gaussian_gain_function(self.changed_durata_pdc_ov,
                mu=self.target_durata_pdc_ov, sig=30)) + \
               (self.gaussian_gain_function(self.changed_arretrato_pdc_ov,
                mu=self.target_arretrato_pdc_ov, sig=30)) + \
               (self.gaussian_gain_function(self.changed_durata_pds,
                mu=self.target_durata_pds, sig=30)) + \
               (self.gaussian_gain_function(self.changed_arretrato_pds,
                mu=self.target_arretrato_pds, sig=30))

        return gain

    def plot_gain_function(self, target_value=60, max_gain=100, function_shape='gaussian'):
        if function_shape == 'linear':
            pass
        elif function_shape == 'gaussian':
            x_values = np.linspace(0, 180, 1000)
            plt.plot(x_values, self.gaussian_gain_function(x_values, mu=target_value, sig=30))
            plt.show()
        return


class Mountain:

    def __init__(self, param_1_dim, param_2_dim):
        filename = cbook.get_sample_data('jacksboro_fault_dem.npz',
                                         asfileobj=False)
        with np.load(filename) as dem:
            z = dem['elevation']
            nrows, ncols = z.shape
            x = np.linspace(dem['xmin'], dem['xmax'], ncols)
            y = np.linspace(dem['ymin'], dem['ymax'], nrows)
            x, y = np.meshgrid(x, y)
            region = np.s_[0:param_1_dim, 0:param_2_dim]
            self.x, self.y, self.z = x[region], y[region], z[region]

    def altitude_function(self, params):
        return self.z[params[0], params[1]]

    def surface_plot_2d(self):
        fig, ax = plt.subplots()
        ax.imshow(self.z, interpolation='nearest')
        plt.show()

    def particle_trajectory(self, particle_data):
        p = particle_data
        iterations = len(particle_data[0])
        dimensions = 3
        trajectory_data = np.empty((dimensions, iterations))
        for i in range(iterations):
            trajectory_data[:, i] = self.x[int(p[0, i]), int(p[1, i])], \
                                    self.y[int(p[0, i]), int(p[1, i])], \
                                    self.z[int(p[0, i]), int(p[1, i])]+10
        return trajectory_data

    def update_trajectories(self, num, trajectories_data, lines):
        for line, data in zip(lines, trajectories_data):
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines

    def surface_plot_3d(self, particles_data):
        iterations = len(particles_data[0][0])
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(270, 45)
        rgb = ls.shade(self.z, cmap=cm.gist_earth, vert_exag=0.1,
                       blend_mode='soft')

        ax.plot_surface(self.x, self.y, self.z*(1-0.5), facecolors=rgb,
                        antialiased=True)

        data = [self.particle_trajectory(particle_data)
                for particle_data in particles_data]
        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1]+10)[0]
                 for dat in data]
        # points = [ax.scatter3D(self.x[int(p[0][1]), int(p[1][1])],
        #                        self.y[int(p[0][1]), int(p[1][1])],
        #                        self.z[int(p[0][1]), int(p[1][1])]+10,
        #                        s=30, c='r') for p in particles_data]

        animate_trajectories = animation.FuncAnimation(
            fig=fig, func=self.update_trajectories, frames=iterations,
            fargs=(data, lines), interval=24, blit=False)
        plt.show()


def pso_mpe(i=10, p=3, iw=.75, cw=.5, sw=.5, v=1):

    mpe = MPE()
    pso = PSO(gain_function=mpe.gain_function, parameters=mpe.params,
              iterations=i, particle_amount=p,
              inertial_weight=iw, cognitive_weight=cw, social_weight=sw,
              verbose=v)
    particles_data, solution_values = pso.iter_particle_swarm()
    mpe.evaluate_solution(solution_values)


def pso_test(s=255, i=100, p=10, iw=.75, cw=.5, sw=.5, v=1):
    
    param_1 = ParameterSampling(0, s-1, s)
    param_2 = ParameterSampling(0, s-1, s)
    params = [param_1, param_2]
    # print(params)
    fnc = Mountain(s, s)
    gain_function = fnc.altitude_function
    pso = PSO(gain_function=gain_function, parameters=params, iterations=i,
              particle_amount=p, inertial_weight=iw, cognitive_weight=cw,
              social_weight=sw, verbose=v)
    particles_data, solution_values = pso.iter_particle_swarm()
    fnc.surface_plot_3d(particles_data)


if __name__ == '__main__':
    # pso_test(i=100, p=10, iw=.5, v=1)
    # pso_mpe(i=100, p=300, iw=0.2, cw=.2, sw=.8, v=1)
    mpe = MPE()
    mpe.plot_gain_function()
