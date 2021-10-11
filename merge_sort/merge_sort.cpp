#include <iostream>
#include <vector>

using namespace std;

vector<int> simple_merge_sort(vector<int> x){
    if(x.size()<=1) return x;
    vector<int> sorted_list;
    int midpoint=-1;
    if(x.size()%2==0){
        midpoint=x.size()/2;
    }else{
        midpoint=(x.size()-1)/2;
    }
    vector<int> left_x={x.begin(),x.begin()+midpoint};
    vector<int> right_x={x.begin()+midpoint,x.end()};
    vector<int> left_sorted=simple_merge_sort(left_x);
    vector<int> right_sorted=simple_merge_sort(right_x);
    //merge phase
    while(left_sorted.size()>0&&right_sorted.size()>0){
        if(left_sorted[0]<right_sorted[0]){
            sorted_list.push_back(left_sorted[0]);
            left_sorted.erase(left_sorted.begin());
        }else{
            sorted_list.push_back(right_sorted[0]);
            right_sorted.erase(right_sorted.begin());
        }
    }
    while(left_sorted.size()>0){
        sorted_list.push_back(left_sorted[0]);
        left_sorted.erase(left_sorted.begin());
    }
    while(right_sorted.size()>0){
        sorted_list.push_back(right_sorted[0]);
        right_sorted.erase(right_sorted.begin());
    }
    return sorted_list;
}

int main(int argc,char* argv[]){
    //generate a list for sorting
    vector<int> x;
    for(int i=10;i>-10;i--) x.push_back(i);
    cout<<"original list:\n";
    for(int i=0;i<x.size();i++) cout<<x[i]<<" ";
    cout<<"\n";
    vector<int> sorted_list=simple_merge_sort(x);
    cout<<"sorted list:\n";
    for(int i=0;i<sorted_list.size();i++) cout<<sorted_list[i]<<" ";
    cout<<"\n";
    return 1;
}

