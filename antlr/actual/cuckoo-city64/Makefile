# goto's performance decreases with march=native
all:
	gcc -O3 -o nogoto nogoto.c city.c cuckoo.c -lrt -lpapi -Wall -Werror -march=native
	gcc -O3 -o goto goto.c city.c cuckoo.c -lrt -lpapi -Wall -Werror
	gcc -O3 -o handopt handopt.c city.c cuckoo.c -lrt -lpapi -Wall -Werror -march=native

clean:
	rm -f *.o goto nogoto handopt
