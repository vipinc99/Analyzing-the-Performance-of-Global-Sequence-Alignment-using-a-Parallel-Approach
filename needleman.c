#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_LEN 1000
#define MATCH 1
#define MISMATCH -1
#define GAP -2
int score(char a, char b) {
if(a==b)
return MATCH;
else
return MISMATCH;
}
int max(int a, int b, int c)
{
int m = a;
if (b > m) m = b;
if (c > m) m = c;
return m;
}
void backtrace(char seq1[], char seq2[], int len1, int len2, int similarity[][MAX_LEN+1])
{
char aligned_seq1[MAX_LEN], aligned_seq2[MAX_LEN];
int i = len1, j = len2, k = 0;
while (i > 0 || j > 0) {
if (i > 0 && j > 0 && similarity[i][j] == similarity[i-1][j-1] + score(seq1[i-1], seq2[j-1])) {
aligned_seq1[k] = seq1[i-1];
aligned_seq2[k] = seq2[j-1];
i--;
j--;
}
else if (i > 0 && similarity[i][j] == similarity[i-1][j] + GAP) {
aligned_seq1[k] = seq1[i-1];
aligned_seq2[k] = '-';
i--;
}
else
{
aligned_seq1[k] = '-';
aligned_seq2[k] = seq2[j-1];
j--;
}
k++;
}
printf("\nGlobal Aligned Sequences:\n");
for (i = k-1; i >= 0; i--) {
printf("%c", aligned_seq1[i]);
}
printf("\n");
for (i = k-1; i >= 0; i--) {
printf("%c", aligned_seq2[i]);
}
printf("\n");
}
int main() {
char seq1[MAX_LEN], seq2[MAX_LEN];
int len1, len2;
int i, j, k;
int similarity[MAX_LEN+1][MAX_LEN+1];
int gap_penalty = GAP;
int match_score, delete_score, insert_score;

printf("Enter first sequence: ");
scanf("%s", seq1);
len1 = strlen(seq1);
printf("Enter second sequence: ");
scanf("%s", seq2);
len2 = strlen(seq2);
//Initialization
for (i = 0; i <= len1; i++)
similarity[i][0] = gap_penalty * i;
for (j = 0; j <= len2; j++)
similarity[0][j] = gap_penalty * j;

// dynamic programming in parallel to calculate similarity matrix
double t1=omp_get_wtime();
int chunk;
for (i = 1; i <= len1; i++) {
#pragma omp parallel default(shared)
{
#pragma omp single
{
chunk=(len1*len2)/(omp_get_num_threads());
}
#pragma omp for private(j, match_score, delete_score, insert_score) schedule(dynamic,chunk)
for (j = 1; j <= len2; j++) {
match_score = similarity[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
delete_score = similarity[i-1][j] + gap_penalty;
insert_score = similarity[i][j-1] + gap_penalty;
similarity[i][j] = max(match_score, delete_score, insert_score);
}
}
}
double t2=omp_get_wtime();

printf("Similarity Matrix:\n    ");
for (j = 0; j < len2; j++)
printf("%3c ", seq2[j]);
printf("\n");
for (i = 0; i <= len1; i++) {
if (i == 0) printf(" ");
else printf("%c ", seq1[i-1]);
for (j = 0; j <= len2; j++)
printf("%3d ", similarity[i][j]);
printf("\n");
}
// backtracing to get optimal alignment sequence
backtrace(seq1, seq2, len1, len2, similarity);
printf("\nChunk size is %d\n",chunk);
printf("\nTime taken to execute Needleman Wunsch algorithm with parallelization is: %lf seconds\n",(t2-t1));
printf("\n");
return 0;
}
