readFile = open("./output.txt")
lines = readFile.readlines()
readFile.close()

fwrite1 = open("./outputs/trainingloss.txt", "w+")
fwrite2 = open("./outputs/trainingacc1.txt", "w+")
fwrite3 = open("./outputs/trainingacc5.txt", "w+")
fwrite4 = open("./outputs/validationloss.txt", "w+")
fwrite5 = open("./outputs/validationacc1.txt", "w+")
fwrite6 = open("./outputs/validationacc5.txt", "w+")

i = 0

for line in lines:
	lol = line.split()

	l = float(lol[4][:-1])
	acc1 = float(lol[8][:-1])
	acc5 = float(lol[11])

	if i % 2 == 0:
		fwrite1.write("{:.6f}".format(l)+"\n")
		fwrite2.write("{:.4f}".format(acc1)+"\n")
		fwrite3.write("{:.4f}".format(acc5)+"\n")
	else:
		fwrite4.write("{:.6f}".format(l)+"\n")
		fwrite5.write("{:.4f}".format(acc1)+"\n")
		fwrite6.write("{:.4f}".format(acc5)+"\n")

	i += 1


fwrite1.close()
fwrite2.close()
fwrite3.close()
fwrite4.close()
fwrite5.close()
fwrite6.close()