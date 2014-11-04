# Copyright 2007-2014 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import numpy as np

import nose.tools as nt
from hyperspy._signals.spectrum import Spectrum
from hyperspy.component import Parameter, Component
from hyperspy.model import Model
from hyperspy.hspy import create_model
from hyperspy.components import Gaussian, Lorentzian, ScalableFixedPattern


def remove_empty_numpy_strings(dic):
    for k, v in dic.iteritems():
        if isinstance(v, dict):
            remove_empty_numpy_strings(v)
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, dict):
                    remove_empty_numpy_strings(vv)
                elif isinstance(vv, np.string_) and len(vv) == 0:
                    vv = ''
        elif isinstance(v, np.string_) and len(v) == 0:
            del dic[k]
            dic[k] = ''


class DummyAxesManager:
    navigation_shape = [1, ]
    navigation_size = 2
    indices = ()

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]


class TestParameterDictionary:

    def setUp(self):
        self.par = Parameter()
        self.par.name = 'asd'

        def ft(x):
            return x * x

        def fit(x):
            return x * x + 1
        self.par.twin_function = ft
        self.par.twin_inverse_function = fit
        self.par._axes_manager = DummyAxesManager()
        self.par._create_array()
        self.par.value = 1
        self.par.std = 0.1
        self.par.store_current_value_in_array()
        self.par.ext_bounded = False
        self.par.ext_force_positive = False

    def test_to_dictionary(self):
        d = self.par.as_dictionary()

        nt.assert_true(d['name'] == self.par.name)
        nt.assert_true(d['_id_name'] == self.par._id_name)
        nt.assert_true(d['map']['values'][0] == 1)
        nt.assert_true(d['map']['std'][0] == 0.1)
        nt.assert_true(d['map']['is_set'][0])
        nt.assert_true(d['value'] == self.par.value)
        nt.assert_true(d['std'] == self.par.std)
        nt.assert_true(d['free'] == self.par.free)
        nt.assert_true(d['_id_'] == id(self.par))
        nt.assert_true(d['_bounds'] == self.par._bounds)
        nt.assert_true(d['ext_bounded'] == self.par.ext_bounded)
        nt.assert_true(
            d['ext_force_positive'] == self.par.ext_force_positive)

    def test_load_dictionary(self):
        d = self.par.as_dictionary()
        p = Parameter()
        _id = p._load_dictionary(d)

        nt.assert_equal(_id, id(self.par))
        nt.assert_true(p.name == self.par.name)
        nt.assert_true(p._id_name == self.par._id_name)
        nt.assert_true(p.map['values'][0] == 1)
        nt.assert_true(p.map['std'][0] == 0.1)
        nt.assert_true(p.map['is_set'][0])
        nt.assert_true(p.value == self.par.value)
        nt.assert_true(p.std == self.par.std)
        nt.assert_true(p.free == self.par.free)
        nt.assert_true(p._bounds == self.par._bounds)

        rn = np.random.random()
        nt.assert_equal(p.twin_function(rn), self.par.twin_function(rn))
        nt.assert_equal(
            p.twin_inverse_function(rn),
            self.par.twin_inverse_function(rn))

    @nt.raises(ValueError)
    def test_invalid_name(self):
        d = self.par.as_dictionary()
        d['_id_name'] = 'newone'
        p = Parameter()
        _id = p._load_dictionary(d)


class TestComponentDictionary:

    def setUp(self):
        self.parameter_names = ['par1', 'par2']
        self.comp = Component(self.parameter_names)
        self.comp.name = 'newname!'
        self.comp._id_name = 'dummy names yay!'
        self.comp._axes_manager = DummyAxesManager()
        self.comp._create_arrays()
        self.comp.par1.value = 2.
        self.comp.par2.value = 5.
        self.comp.par1.std = 0.2
        self.comp.par2.std = 0.5
        self.comp.store_current_parameters_in_map()

    def test_to_dictionary(self):
        d = self.comp.as_dictionary()
        c = self.comp

        nt.assert_equal(c.name, d['name'])
        nt.assert_equal(c._id_name, d['_id_name'])
        nt.assert_false(d['active_is_multidimensional'])
        nt.assert_true(d['active'])
        for ip, p in enumerate(c.parameters):
            nt.assert_equal(p.as_dictionary(), d['parameters'][ip])

        c.active_is_multidimensional = True
        d1 = c.as_dictionary()
        nt.assert_true(d1['active_is_multidimensional'])
        nt.assert_true(d1['_active_array'] == c._active_array)

    def test_load_dictionary(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component(self.parameter_names)
        n._id_name = 'dummy names yay!'
        _ = n._load_dictionary(d)
        nt.assert_equal(c.name, n.name)
        nt.assert_equal(c.active, n.active)
        nt.assert_equal(
            c.active_is_multidimensional,
            n.active_is_multidimensional)

        for pn, pc in zip(n.parameters, c.parameters):
            dn = pn.as_dictionary()
            del dn['_id_']
            dc = pc.as_dictionary()
            del dc['_id_']
            nt.assert_true(dn == dc)

    @nt.raises(ValueError)
    def test_invalid_component_name(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component(self.parameter_names)
        id_dict = n._load_dictionary(d)

    @nt.raises(ValueError)
    def test_invalid_parameter_name(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component([a + 's' for a in self.parameter_names])
        n._id_name = 'dummy names yay!'
        id_dict = n._load_dictionary(d)


class TestModelDictionary:

    def setUp(self):
        s = Spectrum(np.array([1.0, 2, 4, 7, 12, 7, 4, 2, 1]))
        m = create_model(s)
        m._low_loss = (s + 3.0).deepcopy()
        self.model = m

        m.append(Gaussian())
        m.append(Gaussian())
        m.append(ScalableFixedPattern(s * 0.3))
        m[0].A.twin = m[1].A
        m.fit()

    def test_to_dictionary(self):
        m = self.model
        d = m.as_dictionary()

        nt.assert_equal(m.low_loss, d['low_loss'])

        nt.assert_true(np.all(m.chisq.data == d['chisq.data']))

        nt.assert_true(np.all(m.dof.data == d['dof.data']))

        nt.assert_equal(m.spectrum, d['spectrum'])

        nt.assert_equal(
            d['free_parameters_boundaries'],
            m.free_parameters_boundaries)
        nt.assert_equal(d['convolved'], m.convolved)

        for num, c in enumerate(m):
            tmp = c.as_dictionary()
            remove_empty_numpy_strings(tmp)
            nt.assert_equal(d['components'][num]['name'], tmp['name'])
            nt.assert_equal(d['components'][num]['_id_name'], tmp['_id_name'])
        nt.assert_equal(
            d['components'][-1]['_whitelist']['_init_spectrum'], m.spectrum * 0.3)

    def test_load_dictionary(self):
        d = self.model.as_dictionary()
        mn = Model(d)
        # mn = create_model(d)
        mn.append(Lorentzian())
        mn._load_dictionary(d)
        mo = self.model

        nt.assert_true(np.allclose(mo.spectrum.data, mn.spectrum.data))
        nt.assert_true(np.allclose(mo.chisq.data, mn.chisq.data))
        nt.assert_true(np.allclose(mo.dof.data, mn.dof.data))

        nt.assert_true(np.allclose(mn._low_loss.data, mo._low_loss.data))

        nt.assert_equal(
            mn.free_parameters_boundaries,
            mo.free_parameters_boundaries)
        nt.assert_equal(mn.convolved, mo.convolved)
        for i in range(len(mn)):
            nt.assert_equal(mn[i]._id_name, mo[i]._id_name)
            for po, pn in zip(mo[i].parameters, mn[i].parameters):
                nt.assert_true(np.allclose(po.map['values'], pn.map['values']))
                nt.assert_true(np.allclose(po.map['is_set'], pn.map['is_set']))

        nt.assert_true(mn[0].A.twin is mn[1].A)
