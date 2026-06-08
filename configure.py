import argparse, re

makefile_input = "Makefile.in"
makefile_output = "Makefile"

arch_options = ["turing", "ampere", "ada", "hopper", "blackwell"]

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

gpu_options = {"mx450": "turing",
            "mx550": "turing",
            "gtx1630": "turing",
            "gtx1650": "turing",
            "gtx1660": "turing",
            "rtx2060": "turing",
            "rtx2070": "turing",
            "rtx2080": "turing",
            "rtx2080ti": "turing",
            "rtxtitan": "turing",
            "rtx3000": "turing",
            "rtx4000": "turing",
            "rtx5000": "turing",
            "rtx6000": "turing",
            "rtx8000": "turing",
            "t400": "turing",
            "t500": "turing",
            "t600": "turing",
            "t1000": "turing",
            "t1200": "turing",
            "t500": "turing",
            "t4": "turing",
            "t10": "turing",
            "t40": "turing",
            "mx570": "ampere", 
            "rtx2050": "ampere", 
            "rtx3050": "ampere", 
            "rtx3060": "ampere", 
            "rtx3060ti": "ampere", 
            "rtx3070": "ampere", 
            "rtx3070ti": "ampere", 
            "rtx3080": "ampere", 
            "rtx3080ti": "ampere", 
            "rtx3090": "ampere", 
            "rtx3090ti": "ampere", 
            "rtxa1000": "ampere", 
            "rtxa2000": "ampere", 
            "rtxa3000": "ampere", 
            "rtxa4000": "ampere", 
            "rtxa5000": "ampere", 
            "rtxa5500": "ampere", 
            "rtxa6000": "ampere", 
            "a2": "ampere",
            "a10": "ampere",
            "a16": "ampere",
            "a30": "ampere",
            "a40": "ampere",
            "a100": "ampere",
            "a100x": "ampere",
            "a30x": "ampere",
            "rtx4050": "ada", 
            "rtx4060": "ada", 
            "rtx4060ti": "ada", 
            "rtx4070": "ada", 
            "rtx4070ti": "ada", 
            "rtx4080": "ada", 
            "rtx4080super": "ada", 
            "rtx4090": "ada",
            "rtx2000": "ada",
            "rtx4000": "ada",  
            "rtx4500": "ada",  
            "rtx5000": "ada",  
            "rtx5880": "ada",  
            "rtx6000": "ada",
            "rtx500": "ada",
            "rtx1000": "ada",
            "rtx3000": "ada",
            "l4": "ada",
            "l40": "ada", 
            "l40g": "ada", 
            "l40cmx": "ada",
            "h100": "hopper", 
            "h200": "hopper", 
            "h20": "hopper", 
            "h800": "hopper",
            "b100": "blackwell", 
            "b200": "blackwell",
            "rtx5060": "blackwell", 
            "rtx5060ti": "blackwell",  
            "rtx5070": "blackwell", 
            "rtx5070ti": "blackwell",  
            "rtx5080": "blackwell", 
            "rtx5090": "blackwell",                
            }

if __name__ == "__main__":

    # parse command line arguments from config
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None, choices = arch_options, help="select gpu architecture for gencode", type=str.lower)
    parser.add_argument("--gpu", default=None, choices = gpu_options, help="select gpu model for gencode", type=str.lower)

    args = vars(parser.parse_args())

    # collect makefile options
    makefile_options = {}
    makefile_options["GENCODE_FLAGS"] = "-O3 -use_fast_math -Xcompiler -O3 -Wno-deprecated-gpu-targets"
    makefile_options["CXX_FLAGS"] = "-O3 -std=c++23"

    if args["arch"]:
        arch = str(args["arch"]).lower()
        if arch not in arch_options and arch != "none":
            raise Exception("specified GPU architecture not recognised")
        makefile_options["GENCODE_FLAGS"] += " -gencode arch={0},code={1} -Xptxas".format(arch_dict[arch], code_dict[arch])
    elif args["gpu"]:
        gpu = str(args["gpu"]).lower()
        if gpu not in gpu_options.keys():
            raise Exception("specified GPU not listed, select from {0}".format(gpu_options.keys()))
        arch = gpu_options[gpu]
        makefile_options["GENCODE_FLAGS"] += " -gencode arch={0},code={1} -Xptxas".format(arch_dict[arch], code_dict[arch])

    # load template 
    with open(makefile_input, "r") as current_file:
        makefile_template = current_file.read()

    # make replacements
    for key, val in makefile_options.items():
        makefile_template = re.sub(r'@{0}@'.format(key), val, makefile_template)

    # save file
    with open(makefile_output, "w") as current_file:
        current_file.write(makefile_template)