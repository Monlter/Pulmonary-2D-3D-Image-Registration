/*******************************************************************************

    getopt.h - macros to manipulate command line options

    sytax:    ( macros )
        GETOPTSIZ(size)
        GETOPTINT(value)
        GETOPTFLT(value)

	size	    int array of size 3 to hold the values
	value	    int or float variable

    by Guoping Han Rev 1.0, Aug 13, 1999.

*******************************************************************************/

#ifndef GETOPT_H
#define GETOPT_H


#define GETOPTSIZ(size) {			             \
  char *p;                                                   \
  p=(char *)strtok(argv[1], "*x"),  size[0]=atoi(p);         \
  p=(char *)strtok(NULL, "*x"),     size[1]=atoi(p);         \
  p=(char *)strtok(NULL, "*x"),     size[2]=atoi(p);         \
  if (size[2] < 1) size[2]=1;                                \
  argc--; argv++;                                            \
}

#define GETOPTINT(value) {                                   \
  value=atoi(argv[1]);                                       \
  argc--; argv++;                                            \
}

#define GETOPTFLT(value) {                                   \
  value=atof(argv[1]);                                       \
  argc--; argv++;                                            \
}


#endif
