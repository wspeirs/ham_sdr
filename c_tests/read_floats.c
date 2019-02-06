#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Must supply file name\n");
        return -1;
    }

    int fd = open(argv[1], O_RDONLY);

    while(1) {
        float num;
        int res = read(fd, &num, sizeof(float));

        if(res <= 0) {
            break;
        }

        printf("%0.12f\n", num);
    }

    return 0;
}