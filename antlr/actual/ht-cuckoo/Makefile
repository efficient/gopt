all:
	gcc -O3 -o nogoto nogoto.c city.c cuckoo.c -lrt -Wall -Werror 
	gcc -O3 -o goto goto.c city.c cuckoo.c -lrt -Wall -Werror 
	gcc -O3 -o handopt handopt.c city.c cuckoo.c -lrt -Wall -Werror

clean:
	rm -f *.o nogoto handopt goto
