import argparse, re

makefile_input = "Makefile.in"
makefile_output = "Makefile"

gpu_options = ["turing", "ampere", "ada", "hopper", "blackwell"]

arch_dict = {"turing": "compute_75",
            "ampere": "compute_80",
            "ada": "compute_89",
            "hopper": "compute_90",
            "blackwell": "compute_100"}

code_dict = {"turing": "sm_75",
            "ampere": "sm_80",
            "ada": "sm_89",
            "hopper": "sm_90",
            "blackwell": "sm_100"}

if __name__ == "__main__":

    # parse command line arguments from config
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, help="select gpu for gencode")

    args = vars(parser.parse_args())

    # collect makefile options
    makefile_options = {}
    makefile_options = {"GENCODE_FLAGS": "-O3 -use_fast_math -Xcompiler -O3"}

    if args["gpu"]:
        gpu = str(args["gpu"]).lower()
        if gpu not in gpu_options:
            raise Exception("specified GPU not recognised")
        gpu_architecture = arch_dict[gpu]
        gpu_code = code_dict[gpu]
        makefile_options["GENCODE_FLAGS"] += "--gpu-architecture={0} --gpu-code={1} -Xptxas".format(gpu_architecture, gpu_code)

    # load template 
    with open(makefile_input, "r") as current_file:
        makefile_template = current_file.read()

    # make replacements
    for key, val in makefile_options.items():
        makefile_template = re.sub(r'@{0}@'.format(key), val, makefile_template)

    # save file
    with open(makefile_output, "w") as current_file:
        current_file.write(makefile_template)