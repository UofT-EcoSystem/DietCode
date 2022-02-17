def get_time_evaluator_results(kernel, module_data, ctx, number=100, repeat=10,
                               min_repeat_ms=100):
    warmup_evaluator = kernel.time_evaluator(kernel.entry_name, ctx,
                                             number=3, repeat=1,
                                             min_repeat_ms=300)
    warmup_evaluator(*module_data)
    time_evaluator = kernel.time_evaluator(kernel.entry_name, ctx,
                                           number=number, repeat=repeat,
                                           min_repeat_ms=min_repeat_ms)
    return time_evaluator(*module_data).results


def cross_product(argA, argB):
    if not isinstance(argA, list):
        argA = [argA]
    if not isinstance(argB, list):
        argB = [argB]
    args = []
    for a in argA:
        for b in argB:
            if isinstance(a, tuple):
                args.append((*a, *b)) if isinstance(b, tuple) else args.append((*a, b))
            else:
                args.append((a,  *b)) if isinstance(b, tuple) else args.append((a,  b))
    return args
