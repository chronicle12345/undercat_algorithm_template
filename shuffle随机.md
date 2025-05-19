shuffle随机

~~~c++
#include<bits/stdc++.h>
using namespace std;
#define int long long

void solve(){
    vector<int>a(10);
    for(int i=0;i<10;i++)a[i]=i;
    unsigned seed =chrono::system_clock::now().time_since_epoch().count();
    mt19937 g(seed);
    shuffle(a.begin(),a.end(),g);
    for(auto i:a)cout<<i<<" ";
}
signed main(){
    ios::sync_with_stdio(0);cin.tie(0),cout.tie(0);
    int t;cin>>t;while(t--)
    solve();
}
~~~

