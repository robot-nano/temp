import pathlib

import tvm
from tvm import autotvm
import tvm.relay as relay
from tvm.target import Target
from tvm.relay.backend import Runtime
from tvm.relay.backend import Executor


TARGET = tvm.target.target.micro("host")


def test_autotune():
    import tvm.relay as relay
    from tvm.micro.testing.utils import check_tune_log

    runtime = Runtime("crt", {"system-lib": True})

    data = relay.var("data", relay.TensorType((128, 256), "float32"))
    weight = relay.var("weight", relay.TensorType((128, 256), "float"))
    w2 = relay.var("w2", relay.TensorType((10, 128), "float"))
    y = relay.nn.dense(data, weight, out_dtype="float32")
    yw = relay.nn.dense(y, w2, out_dtype="float32")
    f = relay.Function([data, weight, w2], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    main_func = mod['main']
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    target = tvm.target.target.micro("host")
    template_project_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))

    pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
    with pass_context:
        tasks = tvm.autotvm.task.extract_from_program(mod['main'], {}, target)
    print(len(tasks))

    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=template_project_dir,
        project_options={},
    )
    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_options": {"tir.disable_vectorize": True}},
        do_fork=True,
        build_func=tvm.micro.autotvm_build_func,
        runtime=runtime,
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, module_loader=module_loader)

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    tune_log_file = pathlib.Path("crt_autotune.log")
    if tune_log_file.exists():
        tune_log_file.unlink()

    num_trials = 10
    for task in tasks:
        tuner = tvm.autotvm.tuner.GATuner(task)
        tuner.tune(
            n_trial=num_trials,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file(str(tune_log_file)),
                tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M")
            ],
            si_prefix="M",
        )
        assert tuner.best_flops > 0

    with pass_context:
        lowered = tvm.relay.build(mod, target=TARGET, runtime=runtime)

    out_dir = "/home/wserver/ws/TVM/temp/python/out"
    project = tvm.micro.generate_project(out_dir, lowered, out_dir + "/project")
    project.build()


if __name__ == "__main__":
    test_autotune()
