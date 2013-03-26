import nose.tools
import numpy
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.graph import make_graph_from_route_collection

@nose.tools.istest
def make_graph_from_blink_model():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 3)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    dg = make_graph_from_route_collection(m.route_collection)
    initial_state = '3_0_0_0'
    assert dg.has_edge('0_1_1_1', '0_0_1_2')
    r_id_list = []
    for route_id in dg.iter_successors(initial_state, depth=100):
        print route_id
        r_id_list.append(route_id)
    assert ('0_1_1_1__0_0_1_2' in r_id_list),\
            '0_1_1_1__0_0_1_2 not in iter_successors list.'
    for n in dg.iter_neighbors(initial_state):
        print "level 1:", n
        for n2 in dg.iter_neighbors(n):
            print "level 2:", n2
            for n3 in dg.iter_neighbors(n2):
                print "level 3:", n3
