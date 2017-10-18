with open("datalie.txt") as file:
    for line in file:
        line = line.rstrip()
        if line:
            print line