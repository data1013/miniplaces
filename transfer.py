fread = open("derpderp.rtf", "r") 
fwrite1 = open("trainingloss.txt", "w+")
fwrite2 = open("trainingacc1.txt", "w+")
fwrite3 = open("trainingacc5.txt", "w+")
fwrite4 = open("validationloss.txt", "w+")
fwrite5 = open("validationacc1.txt", "w+")
fwrite6 = open("validationacc5.txt", "w+")

def find_between(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        print "ERROR"

for line in fread:
    if "Training Loss" in line:
        fwrite1.write(find_between(line, "Training Loss= ", ",")+"\n")
        fwrite2.write(find_between(line, "Top1 = ", ",")+"\n")
        fwrite3.write(find_between(line, "Top5 = ", "\uc0")+"\n")
    if "Validation Loss" in line:
        fwrite4.write(find_between(line, "Loss= ", ",")+"\n")
        fwrite5.write(find_between(line, "Top1 = ", ",")+"\n")
        fwrite6.write(find_between(line, "Top5 = ", "\uc0")+"\n")

fread.close()
fwrite1.close()
fwrite2.close()
fwrite3.close()
fwrite4.close()
fwrite5.close()
fwrite6.close()