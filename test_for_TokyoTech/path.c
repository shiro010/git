#include <stdio.h>

int t = 0;
int d = 0;

struct Node {
    int x;
    int y;
    struct Node* p;
};

int is_visited(struct Node n){
    struct Node* n_curr = n.p;
    while(n_curr != NULL){
        n_curr = n_curr->p;
        if (n_curr == NULL) return 0;
        if (n_curr->x == n.x && n_curr->y == n.y) return 1;
    }
    return 0;
}

int count_path(struct Node n, int N){
    int dx[] = {0,0,-1,1};
    int dy[] = {-1,1,0,0};
    int res = 0;
    if(is_visited(n)) return 0;
    if (n.x == N && n.y == N) return 1;
    for (int i = 0; i < 4; i++){
        int x = n.x + dx[i];
        int y = n.y + dy[i];
        if(x < 0 || y < 0 || x > N || y > N) continue;
        struct Node n_new;
        n_new.x = x; n_new.y = y; n_new.p = &n;
        res += count_path(n_new, N);
        t++; d++;
        if(t<5)printf("%d %d %d\n", x, y, d);
    }
    
    return res;
}

int main() {
    struct Node n;
    n.x = 0; n.y = 0; n.p = NULL;
    printf("%d\n", count_path(n,2));
    return 0;
}