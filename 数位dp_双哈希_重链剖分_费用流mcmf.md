## 加速
~~~c++
#pragma GCC optimize(3,"Ofast","inline")
#pragma g++ optimize(3,"Ofast","inline")
~~~
## 数位dp
~~~c++
/*如果一个正整数每一个数位都是 互不相同 的，我们称它是 特殊整数 。
给你一个 正 整数 n ，请你返回区间 [1, n] 之间特殊整数的数目。*/
class Solution {
public:
    int countSpecialNumbers(int n) {
        auto s = to_string(n);
        int m = s.length(), memo[m][1 << 10];
        memset(memo, -1, sizeof(memo)); // -1 表示没有计算过
        function<int(int, int, bool, bool)> f = [&](int i, int mask, bool is_limit, bool is_num) -> int {
            if (i == m)
                return is_num; // is_num 为 true 表示得到了一个合法数字
            if (!is_limit && is_num && memo[i][mask] != -1)
                return memo[i][mask];
            int res = 0;
            if (!is_num) // 可以跳过当前数位
                res = f(i + 1, mask, false, false);
            int up = is_limit ? s[i] - '0' : 9; // 如果前面填的数字都和 n 的一样，那么这一位至多填数字 s[i]（否则就超过 n 啦）
            for (int d = 1 - is_num; d <= up; ++d) // 枚举要填入的数字 d
                if ((mask >> d & 1) == 0) // d 不在 mask 中
                    res += f(i + 1, mask | (1 << d), is_limit && d == up, true);
            if (!is_limit && is_num)
                memo[i][mask] = res;
            return res;
        };
        return f(0, 0, true, false);
    }
};
~~~
## dfs序求lca
~~~c++
#include <bits/stdc++.h>
using namespace std;
constexpr int N = 5e5 + 5;
int n, m, R, dn, dfn[N], mi[19][N];
vector<int> e[N];
int get(int x, int y) {return dfn[x] < dfn[y] ? x : y;}
void dfs(int id, int f) {
  mi[0][dfn[id] = ++dn] = f;
  for(int it : e[id]) if(it != f) dfs(it, id); 
}
int lca(int u, int v) {
  if(u == v) return u;
  if((u = dfn[u]) > (v = dfn[v])) swap(u, v);
  int d = __lg(v - u++);
  return get(mi[d][u], mi[d][v - (1 << d) + 1]);
}
int main() {
  scanf("%d %d %d", &n, &m, &R);
  for(int i = 2, u, v; i <= n; i++) {
    scanf("%d %d", &u, &v);
    e[u].push_back(v), e[v].push_back(u);
  }
  dfs(R, 0);
  for(int i = 1; i <= __lg(n); i++)
    for(int j = 1; j + (1 << i) - 1 <= n; j++)
        mi[i][j] = get(mi[i - 1][j], mi[i - 1][j + (1 << i - 1)]);
  for(int i = 1, u, v; i <= m; i++) scanf("%d %d", &u, &v), printf("%d\n", lca(u, v));
}
~~~
## tarjan求lca
~~~c++
#include<bits/stdc++.h>
#define N 500005
using namespace std;
int n,m,s,dep[N],id[N];bool vst[N];
struct query{
	int id,x,y,lca;
}q[N];
struct data{
	int id,num;
};
vector<int>es[N];
vector<data>qes[N];
inline int find(int i){return i==id[i]?i:id[i]=find(id[i]);}
inline void unite(int u,int v){int a=find(u),b=find(v);if(dep[a]>dep[b])id[a]=b;else id[b]=a;}
inline void tarjan(int x){
	id[x]=x;vst[x]=true;
	for(int i=0;i<es[x].size();i++)
		if(!vst[es[x][i]]){
			dep[es[x][i]]=dep[x]+1;
			tarjan(es[x][i]);
			unite(es[x][i],x);
		} 
	for(int i=0;i<qes[x].size();i++)if(vst[qes[x][i].num])q[qes[x][i].id].lca=find(qes[x][i].num);
}
int main(){
	scanf("%d%d%d",&n,&m,&s);
	for(int i=1,x,y;i<n;i++)scanf("%d%d",&x,&y),es[x].push_back(y),es[y].push_back(x);
	for(int i=1;i<=m;i++){
		scanf("%d%d",&q[i].x,&q[i].y);
		q[i].id=i;
		qes[q[i].x].push_back((data){i,q[i].y});
		qes[q[i].y].push_back((data){i,q[i].x});
	}
	tarjan(s);
	for(int i=1;i<=m;i++)printf("%d\n",q[i].lca);
}
~~~
## 字符串双哈希
~~~c++
struct Shash{
    #define ll long long
    const ll base[2]={29,31};
    const ll hashmod[2]={(ll)1e9,998244353};
    array<vector<ll>,2>hsh,pwMod;
    void init(string &s){
        int n=s.size();s=' '+s;
        hsh[0].resize(n+1),hsh[1].resize(n+1);
        pwMod[0].resize(n+1),pwMod[1].resize(n+1);
        for(int i=0;i<2;i++){
            pwMod[i][0]=1;
            for(int j=1;j<=n;j++){
                pwMod[i][j]=pwMod[i][j-1]*base[i]%hashmod[i];
                hsh[i][j]=(hsh[i][j-1]*base[i]+s[j])%hashmod[i];
            }
        }
    }
    pair<ll,ll>get(int l,int r){
        pair<ll,ll>ans;
        ans.fi=(hsh[0][r]-hsh[0][l-1]*pwMod[0][r-l+1])%hashmod[0];
        ans.se=(hsh[1][r]-hsh[1][l-1]*pwMod[1][r-l+1])%hashmod[1];
        ans.fi=(ans.fi+hashmod[0])%hashmod[0];
        ans.se=(ans.se+hashmod[1])%hashmod[1];
        return ans;
    }
    bool same(int la,int ra,int lb,int rb){
        return get(la,ra)==get(lb,rb);
    }
};
~~~
## 重链剖分
~~~c++
/*如题，已知一棵包含 N 个结点的树（连通且无环），每个节点上包含一个数值，需要支持以下操作：
1 x y z，表示将树从 x 到 y 结点最短路径上所有节点的值都加上z
2 x y，表示求树从x 到y 结点最短路径上所有节点的值之和
3 x z，表示将以x 为根节点的子树内所有节点值都加上z
4 x 表示求以 x 为根节点的子树内所有节点值之和*/
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int N=1e5+10;
vector<int>g[N];
int mod;
int dep[N],top[N],fa[N],son[N],sz[N],a[N],na[N],id[N],num;
void dfs1(int u,int f){
    sz[u]=1;fa[u]=f;
    dep[u]=dep[f]+1;
    for(auto i:g[u]){
        if(i==f)continue;
        dfs1(i,u);
        sz[u]+=sz[i];
        if(sz[son[u]]<sz[i])son[u]=i;
    }
}
void dfs2(int u,int t){
    id[u]=++num;na[num]=a[u];
    top[u]=t;
    if(!son[u])return;
    dfs2(son[u],t);
    for(auto i:g[u]){
         if(i!=fa[u]&&i!=son[u])dfs2(i,i);
    }
}
struct segtree{
    struct node{
        int l,r,sum,lazy;
    };
    vector<node>tr;
    segtree(int n):tr(n<<2){};
    void pushdown(int u){
        if(tr[u].lazy){
            tr[u<<1].lazy+=tr[u].lazy;
            tr[u<<1|1].lazy+=tr[u].lazy;
            tr[u<<1].sum=(tr[u<<1].sum+tr[u].lazy*(tr[u<<1].r-tr[u<<1].l+1))%mod;
            tr[u<<1|1].sum=(tr[u<<1|1].sum+tr[u].lazy*(tr[u<<1|1].r-tr[u<<1|1].l+1))%mod;
            tr[u].lazy=0;
        }
    }
    void pushup(int u){
        tr[u].sum=(tr[u<<1].sum+tr[u<<1|1].sum)%mod;
    }
    void build(int u,int l,int r){
        tr[u]={l,r};
        if(l==r){
            tr[u].sum=na[l];
            return;
        }
        int mid=(l+r)>>1;
        build(u<<1,l,mid);
        build(u<<1|1,mid+1,r);
        pushup(u);
    }
    void update(int u,int l,int r,int v){
        if(l<=tr[u].l&&tr[u].r<=r){
            tr[u].sum=(tr[u].sum+v*(tr[u].r-tr[u].l+1))%mod;
            tr[u].lazy+=v;
        }else{
            pushdown(u);
            int mid=(tr[u].l+tr[u].r)>>1;
            if(l<=mid)update(u<<1,l,r,v);
            if(r>mid)update(u<<1|1,l,r,v);
            pushup(u);
        }
    }
    int query(int u,int l,int r){
        if(l<=tr[u].l&&tr[u].r<=r){
            return tr[u].sum%mod;
        }else{
            pushdown(u);
            int ans=0;
            int mid=(tr[u].l+tr[u].r)>>1;
            if(l<=mid)ans=query(u<<1,l,r);
            if(r>mid)ans+=query(u<<1|1,l,r);
            return ans%mod;
        }   
    }
};
segtree tree(N);
void update(int x,int y,int v){
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]])swap(x,y);
        tree.update(1,id[top[x]],id[x],v);
        x=fa[top[x]];
    }
    if(dep[x]>dep[y])swap(x,y);
    tree.update(1,id[x],id[y],v);
}
int query(int x,int y){
    int ans=0;
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]])swap(x,y);
        ans=(ans+tree.query(1,id[top[x]],id[x]))%mod;
        x=fa[top[x]];
    }
    if(dep[x]>dep[y])swap(x,y);
    return (ans+tree.query(1,id[x],id[y]))%mod;
}
signed main(){
    ios::sync_with_stdio(0);cin.tie(0),cout.tie(0);
    int n,m,r;cin>>n>>m>>r>>mod;
    for(int i=1;i<=n;i++)cin>>a[i];
    for(int i=1,u,v;i<n;i++){
        cin>>u>>v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs1(r,0);dfs2(r,r);
    tree.build(1,1,n);
    while(m--){
        int op,x,y,z;cin>>op;
        if(op==1){
            cin>>x>>y>>z;
            update(x,y,z);
        }else if(op==2){
            cin>>x>>y;
            cout<<query(x,y)<<'\n';
        }else if(op==3){
            cin>>x>>z;
            tree.update(1,id[x],id[x]+sz[x]-1,z);
        }else{
            cin>>x;
            cout<<tree.query(1,id[x],id[x]+sz[x]-1)<<'\n';
        }
    }
}
~~~
## 优化网络流mcmf（费用流/网络流）
~~~c++
#include<bits/stdc++.h>
#define int long long
namespace flow { // }{{{
    #define UP(i,s,e) for(auto i=s; i<e; ++i)
    #define ll long long
    constexpr int VV = 5200, EE = 80000;
    constexpr int V = VV + 1, E = EE;
    struct Edge {
        int to;
        ll lf, cost;
        int nxt;
    } es[E * 2 + 4];
    int ecnt, head[V], fe[V], fa[V], tim, mark[V];
    ll sumflow, sumcost, maxf, minc, pi[V];
    void init() {
        std::fill(mark, mark + V, 0);
        std::fill(head, head + V, 0);
        //补充初始化
        memset(es,0,sizeof es);memset(pi,0,sizeof pi);
        memset(fa,0,sizeof fa);memset(fe,0,sizeof fe);
        //end
        tim = ecnt = 2;
        sumflow = sumcost = minc = maxf = 0;
    }
    void addflow(int s, int t, ll f, ll c) {
        s++, t++;
        es[ecnt] = {t, f, c, head[s]};
        head[s] = ecnt++;
        es[ecnt] = {s, 0, -c, head[t]};
        head[t] = ecnt++;
        sumflow += f;
        sumcost += std::abs(c);
    }
    void mktree(int x, int fre) {
        mark[x] = 1;
        fe[x] = fre;
        fa[x] = es[fre ^ 1].to;
        for (int i = head[x]; i; i = es[i].nxt) {
            if (es[i].lf == 0 || mark[es[i].to] == 1)
                continue;
            mktree(es[i].to, i);
        }
    }
    ll getpi(int x) {
        if (mark[x] == tim)
            return pi[x];
        mark[x] = tim;
        return pi[x] = getpi(fa[x]) + es[fe[x]].cost;
    }
    ll pushflow(int e) {
        int rt = es[e].to, lca = es[e ^ 1].to;
        tim++;
        while (rt) {
            mark[rt] = tim;
            rt = fa[rt];
        }
        while (mark[lca] != tim) {
            mark[lca] = tim;
            lca = fa[lca];
        }
        static std::vector<int> topush;
        topush.clear();
        topush.push_back(e);
        ll df = es[e].lf;
        int todel = e, dir = -1;
        UP(j, 0, 2) for (int i = es[e ^ !j].to; i != lca; i = fa[i]) {
            if (es[fe[i]^j].lf < df) {
                todel = fe[i] ^ j;
                dir = !j;
                df = es[todel].lf;
            }
            topush.push_back(fe[i]^j);
        }
        ll dcst = 0;
        if (df)
            for (int i : topush) {
                dcst += es[i].cost * df;
                es[i].lf -= df;
                es[i ^ 1].lf += df;
            }
        if (todel == e)
            return dcst;
        int laste = e ^ dir;
        for (int i = es[e ^ dir].to; i != es[todel ^ dir].to;) {
            int ii = fa[i];
            std::swap(fe[i], laste);
            laste ^= 1;
            fa[i] = es[fe[i] ^ 1].to;
            mark[i] = 0;
            i = ii;
        }

        return dcst;
    }
    void mcmf(int s, int t) {
        ll sfl = sumflow + 1, scs = sumcost + 1;
        addflow(t, s, sfl, -scs);
        mktree(t, 0);
        int i = 2, j = 2;
        do {
            if (es[i].lf && getpi(es[i ^ 1].to) - getpi(es[i].to) + es[i].cost < 0) {
                minc += pushflow(j = i);
            }
            i = (i == ecnt - 1 ? 2 : i + 1);
        } while (i != j);
        maxf = es[ecnt - 1].lf;
        minc += maxf * scs;
    }
    #undef UP
    #undef ll
} // {}}}
namespace m { // }{{{
    int in, im, is, it;
    void work() {
        scanf("%lld%lld%lld%lld", &in, &im, &is, &it);
        is--;it--;
        flow::init();
        for(int i=0;i<im;i++){
            int iu, iv, ic;
            scanf("%lld%lld%lld", &iu, &iv, &ic);
            int c=0;
            //scanf("%lld",&c);//费用
            iu--;
            iv--;
            flow::addflow(iu, iv, ic, c);
        }
        flow::mcmf(is, it);
        printf("%lld\n", flow::maxf);//最大流
        //printf("%lld %lld\n", flow::maxf,flow::minc);//最小费用最大流
    }
} // {}}}
signed main() {
    m::work();
}
~~~

## 费用流板子2

~~~c++
struct edg{int id,s,e;long long vol,cost;};
struct MCMF_spfa_EK{
    const long long inf=4e13,inf2=3e9;
    vector<edg>einfo;vector<vector<int>>es;int tim=0,etot=-1;long long maxf=0,ans=0;//vis:update dis
    vector<int>vis,inq,lst_e;vector<long long>mxf,dis;
    void add(int u,int v,long long vol,long long cost){
        es[u].push_back(++etot);einfo.push_back({etot,u,v,vol,cost});
        es[v].push_back(++etot);einfo.push_back({etot,v,u,0,-cost});}
    bool spfa(int S,int T){
        queue<int>Q;Q.push(S);dis[S]=0;inq[S]=1,vis[S]=++tim;mxf[S]=inf;
        for(;!Q.empty();Q.pop()){int x=Q.front();inq[x]=0;vis[x]=tim;
            for(auto eid:es[x]){edg &k=einfo[eid];int des=k.e;if(vis[des]!=tim)dis[des]=inf;
                if(!k.vol||k.cost+dis[x]>=dis[des])continue;vis[des]=tim;dis[des]=dis[x]+k.cost;
                mxf[des]=min(mxf[x],k.vol);lst_e[des]=eid;if(!inq[des])inq[des]=1,Q.push(des);}}
        if(vis[T]!=tim)dis[T]=inf;return dis[T]!=inf;}
    void Solve(int S,int T){while(spfa(S,T)){for(int x=T,k;x!=S;x=einfo[k].s)//最小费用最大流//最大:边权全反
  //void Solve(int S,int T){while(spfa(S,T)&&dis[T]<0){for(int x=T,k;x!=S;x=einfo[k].s)//最小费用可行流
        k=lst_e[x],einfo[k].vol-=mxf[T],einfo[k^1].vol+=mxf[T];maxf+=mxf[T];ans+=(long long)(dis[T])*mxf[T];}} 
    void ini(int n){n++;vis.resize(n,0);inq=lst_e=vis;dis.resize(n);mxf=dis;es.resize(n);}
};
    long long maximumValueSum(vector<vector<int>>& board) {
        MCMF_spfa_EK A;int n=board.size(),m=board[0].size();
        int Bas=(n+1)*(m+1),S=0,B=Bas+1,T=B+1;A.ini(Bas+5);
        A.add(S,B,3,0);
        for(int i=1;i<=n;i++)A.add(B,i,1,0);
        for(int i=1;i<=m;i++)A.add(i+n,T,1,0);
        for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)
        {
            int p=(i-1)*m+j+n+m;
            A.add(i,p,1,-board[i-1][j-1]);
            A.add(p,j+n,1,0);
        }
        A.Solve(S,T);return -A.ans;
    }
~~~

## 数剖（jly板子）

~~~c++
#include <bits/stdc++.h>

using u32 = unsigned;
using i64 = long long;
using u64 = unsigned long long;
struct HLD {
    int n;
    std::vector<int> siz, top, dep, parent, in, out, seq;
    std::vector<std::vector<int>> adj;
    int cur;
    
    HLD() {}
    HLD(int n) {
        init(n);
    }
    void init(int n) {
        this->n = n;
        siz.resize(n);
        top.resize(n);
        dep.resize(n);
        parent.resize(n);
        in.resize(n);
        out.resize(n);
        seq.resize(n);
        cur = 0;
        adj.assign(n, {});
    }
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void work(int root = 0) {
        top[root] = root;
        dep[root] = 0;
        parent[root] = -1;
        dfs1(root);
        dfs2(root);
    }
    void dfs1(int u) {
        if (parent[u] != -1) {
            adj[u].erase(std::find(adj[u].begin(), adj[u].end(), parent[u]));
        }
        
        siz[u] = 1;
        for (auto &v : adj[u]) {
            parent[v] = u;
            dep[v] = dep[u] + 1;
            dfs1(v);
            siz[u] += siz[v];
            if (siz[v] > siz[adj[u][0]]) {
                std::swap(v, adj[u][0]);
            }
        }
    }
    void dfs2(int u) {
        in[u] = cur++;
        seq[in[u]] = u;
        for (auto v : adj[u]) {
            top[v] = v == adj[u][0] ? top[u] : v;
            dfs2(v);
        }
        out[u] = cur;
    }
    int lca(int u, int v) {
        while (top[u] != top[v]) {
            if (dep[top[u]] > dep[top[v]]) {
                u = parent[top[u]];
            } else {
                v = parent[top[v]];
            }
        }
        return dep[u] < dep[v] ? u : v;
    }
    
    int dist(int u, int v) {
        return dep[u] + dep[v] - 2 * dep[lca(u, v)];
    }
    
    int jump(int u, int k) {
        if (dep[u] < k) {
            return -1;
        }
        
        int d = dep[u] - k;
        
        while (dep[top[u]] > d) {
            u = parent[top[u]];
        }
        
        return seq[in[u] - dep[u] + d];
    }
    
    bool isAncester(int u, int v) {
        return in[u] <= in[v] && in[v] < out[u];
    }
    
    int rootedParent(int u, int v) {
        std::swap(u, v);
        if (u == v) {
            return u;
        }
        if (!isAncester(u, v)) {
            return parent[u];
        }
        auto it = std::upper_bound(adj[u].begin(), adj[u].end(), v, [&](int x, int y) {
            return in[x] < in[y];
        }) - 1;
        return *it;
    }
    
    int rootedSize(int u, int v) {
        if (u == v) {
            return n;
        }
        if (!isAncester(v, u)) {
            return siz[v];
        }
        return n - siz[rootedParent(u, v)];
    }
    
    int rootedLca(int a, int b, int c) {
        return lca(a, b) ^ lca(b, c) ^ lca(c, a);
    }
};
~~~

