//uva11461
#include<iostream>
#include<cmath>
using namespace std;
int main(){
  int a,b;
  while(cin>>a>>b&&(a*b)){
    int ia=sqrt(a),ib=sqrt(b);
    if((ia+1)*(ia+1)==a)ia++;
    if((ib+1)*(ib+1)==b)ib++;
    int ans=ib-ia+(ia*ia==a);
    cout<<ans<<endl;
  }
  return 0;
}
