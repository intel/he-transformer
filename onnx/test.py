import onnx
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng
import numpy as np
import argparse


def run(FLAGS):

    onnx_protobuf = onnx.load('cryptonets.onnx')

    #print('model.graph.input[0]', onnx_protobuf.graph.input[0])
    #onnx_protobuf.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '10'
    #print('model.graph.input[0]', onnx_protobuf.graph.input[0])

    #print('model.graph.output[0]', onnx_protobuf.graph.output[0])
    #onnx_protobuf.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '10'
    #print('model.graph.output[0]', onnx_protobuf.graph.output[0])

    #onnx.save(onnx_protobuf, 'cryptonets-fixed.onnx')

    #onnx_protobuf = onnx.load('cryptonets-fixed.onnx')

    #print('onnx_protobuf', onnx_protobuf)

    ng_function = import_onnx_model(onnx_protobuf)

    print(ng_function)

    picture = np.ones([16, 28, 28, 1], dtype=np.float32)

    # Run on interpreter
    int_runtime = ng.runtime(backend_name='INTERPRETER')
    int_cryptonets = int_runtime.computation(ng_function)
    int_pred = int_cryptonets(picture)
    print('int_pred', int_pred)

    # Run on he_seal
    he_runtime = ng.runtime(backend_name='HE_SEAL')

    config = {}
    if FLAGS.encryption_parameters != '':
        config['encryption_parameters'] = FLAGS.encryption_parameters

    config['Parameter_8'] = 'encrypt,packed'
    print('config', config)

    he_runtime.set_config(config)
    he_cryptonets = he_runtime.computation(ng_function)
    he_pred = he_cryptonets(picture)
    print('he_pred', he_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        default="",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )

    FLAGS, unparsed = parser.parse_known_args()
    run(FLAGS)
