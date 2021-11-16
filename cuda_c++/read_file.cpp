#include "read_file.h"



void read_file(char * str_infolder)
{
    char s[100];
    _finddata_t file;
    int k;
    intptr_t HANDLE;
    FILE* fp1; 
    if (fopen_s(&fp1, "file_name.txt", "w") != 0) 
    {
        printf("加载txt文件出错\n");
    }
    strcpy(s,str_infolder);
    strcat(s,"*");
    printf("%s\n", s);
    k = HANDLE = _findfirst(s, &file);
    while (k != -1)
    {
        
        if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
        {
            k = _findnext(HANDLE, &file);
            continue;
        }
        fprintf(fp1, "%s\n", file.name);
        k = _findnext(HANDLE, &file);
        
    }
    _findclose(HANDLE);
    fclose(fp1);

    return ;
}
