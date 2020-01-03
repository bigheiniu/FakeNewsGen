from examples.pplm_clickbait.run_pplm import run_pplm_example_file
# it should only keep the title
file_name = "/home/yichuan/huggingface_transformer/MyExample/data/political/train.txt"
run_pplm_example_file(
    file_path=file_name,
    num_samples=2,
    discrim='clickbait',
    class_label='clickbait',
    length=100,
    stepsize=0.05,
    sample=True,
    num_iterations=10,
    gamma=1,
    gm_scale=0.9,
    kl_scale=0.02
    # verbosity='regular'
)

run_pplm_example_file(
    file_path=file_name,
    num_samples=2,
    discrim='clickbait',
    class_label='non_clickbait',
    length=100,
    stepsize=0.05,
    sample=True,
    num_iterations=10,
    gamma=1,
    gm_scale=0.9,
    kl_scale=0.02
    # verbosity='regular'
)


