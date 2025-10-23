

input_file = "temp_klein.txt" # file using the comma-separated list from the pdf.
output_file = "filtration_klein_bottle.txt" # outputs the filtration in the same format as given in the TD
generate_n_spheres_balls = True

from itertools import combinations

def all_combinations(lst):
    result = []
    n = len(lst)
    for k in range(1, n+1):
        for combo in combinations(lst, k):
            result.append(list(combo))  # keep as a list of numbers
    return result


if not generate_n_spheres_balls:
    with open(input_file, "r+") as file:
        content = file.read()
        print(content)
        sep = content.split(",")
        print(sep)

        for i in range(len(sep)):
            sep[i] = sep[i].strip()
            sep[i] = sep[i].strip(".")
        print(sep)
        with open(output_file, "w+") as file:
            space_split = []
            for val in sep:
                space_split.append(' ' + ' '.join(str(val)))

            print(space_split)
            for i, val in enumerate(sep):
                file.write(f"{i}.0 {len(str(val).strip())-1}{space_split[i]}\n")

else:
    for N in range(0, 10):
        sphere_filename = f"{N}-sphere.txt"
        ball_filename = f"{N}-ball.txt"

        vertices_ball = [k for k in range(1, N + 2)]
        vertices_sphere = [k for k in range(1, N + 3)]

        all_combs_balls = all_combinations(vertices_ball)
        all_combs_sphere = all_combinations(vertices_sphere)[:-1]  # remove top-dimensional simplex

        # write sphere
        with open(sphere_filename, "w+") as out1:
            for i, val in enumerate(all_combs_sphere):
                spaced = ' ' + ' '.join(map(str, val))
                out1.write(f"{i}.0 {len(val) - 1}{spaced}\n")

        # write ball
        with open(ball_filename, "w+") as out2:
            for i, val in enumerate(all_combs_balls):
                spaced = ' ' + ' '.join(map(str, val))
                out2.write(f"{i}.0 {len(val) - 1}{spaced}\n")


