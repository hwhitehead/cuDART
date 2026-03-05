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

gpu_options = {"MX450": "turing",
            "MX550": "turing",
            "GTX1630": "turing",
            "GTX1650": "turing",
            "GTX1660": "turing",
            "RTX2060": "turing",
            "RTX2070": "turing",
            "RTX2080": "turing",
            "RTX2080Ti": "turing",
            "RTXTitan": "turing",
            "RTX3000": "turing",
            "RTX4000": "turing",
            "RTX5000": "turing",
            "RTX6000": "turing",
            "RTX8000": "turing",
            "T400": "turing",
            "T500": "turing",
            "T600": "turing",
            "T1000": "turing",
            "T1200": "turing",
            "T500": "turing",
            "T4": "turing",
            "T10": "turing",
            "T40": "turing",
            "MX570": "ampere", 
            "RTX2050": "ampere", 
            "RTX3050": "ampere", 
            "RTX3060": "ampere", 
            "RTX3060Ti": "ampere", 
            "RTX3070": "ampere", 
            "RTX3070Ti": "ampere", 
            "RTX3080": "ampere", 
            "RTX3080Ti": "ampere", 
            "RTX3090": "ampere", 
            "RTX3090Ti": "ampere", 
            "RTXA1000": "ampere", 
            "RTXA2000": "ampere", 
            "RTXA3000": "ampere", 
            "RTXA4000": "ampere", 
            "RTXA5000": "ampere", 
            "RTXA5500": "ampere", 
            "RTXA6000": "ampere", 
            "A2": "ampere",
            "A10": "ampere",
            "A16": "ampere",
            "A30": "ampere",
            "A40": "ampere",
            "A100": "ampere",
            "A100X": "ampere",
            "A30X": "ampere",
            "RTX4050": "ada", 
            "RTX4060": "ada", 
            "RTX4060Ti": "ada", 
            "RTX4070": "ada", 
            "RTX4070Ti": "ada", 
            "RTX4080": "ada", 
            "RTX4080Super": "ada", 
            "RTX4090": "ada",
            "RTX2000": "ada",
            "RTX4000": "ada",  
            "RTX4500": "ada",  
            "RTX5000": "ada",  
            "RTX5880": "ada",  
            "RTX6000": "ada",
            "RTX500": "ada",
            "RTX1000": "ada",
            "RTX3000": "ada",
            "L4": "ada",
            "L40": "ada", 
            "L40G": "ada", 
            "L40CNX": "ada",
            "H100": "hopper", 
            "H200": "hopper", 
            "H20": "hopper", 
            "H800": "hopper",
            "B100": "blackwell", 
            "B200": "blackwell",
            "RTX5060": "blackwell", 
            "RTX5060Ti": "blackwell",  
            "RTX5070": "blackwell", 
            "RTX5070Ti": "blackwell",  
            "RTX5080": "blackwell", 
            "RTX5090": "blackwell",                
            }

if __name__ == "__main__":

    # parse command line arguments from config
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None, choices = arch_options, help="select gpu architecture for gencode", type=str.lower)
    parser.add_argument("--gpu", default=None, choices = gpu_options, help="select gpu model for gencode", type=str)

    args = vars(parser.parse_args())

    # collect makefile options
    makefile_options = {}
    makefile_options = {"GENCODE_FLAGS": "-O3 -use_fast_math -Xcompiler -O3 -Wno-deprecated-gpu-targets"}

    if args["arch"]:
        arch = str(args["arch"]).lower()
        if arch not in arch_options and arch != "none":
            raise Exception("specified GPU architecture not recognised")
        makefile_options["GENCODE_FLAGS"] += " -gencode arch={0},code={1} -Xptxas".format(arch_dict[arch], code_dict[arch])
    elif args["gpu"]:
        gpu = str(args["gpu"])
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