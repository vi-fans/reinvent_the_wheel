#include <iostream>
#include <vector>

using namespace std;

vector<int> simple_merge_sort(vector<int> x){
    if(x.size()==1) return x;
    vector<int> sorted_list;
    int midpoint=-1,mid_value;
    if(x.size()%2==0){
        midpoint=x.size()/2;
        mid_value=x[midpoint];
    }else{
        midpoint=(x.size()-1)/2;
        mid_value=x[midpoint+1];
    }
    vector<int> left_x;
    vector<int> right_x;
    for(int i=0;i<x.size();i++){
        if(x[i]<=mid_value){
            left_x.push_back(x[i]);
        }else right_x.push_back(x[i]);
    }
    vector<int> left_sorted=simple_merge_sort(left_x);
    vector<int> right_sorted=simple_merge_sort(right_x);
    sorted_list=left_sorted;
    sorted_list.insert(sorted_list.end(),right_sorted.begin(),right_sorted.end());
    return sorted_list;
}

int main(int argc,char* argv[]){
    //generate a list for sorting
    vector<int> x;
    for(int i=100;i>-100;i--) x.push_back(i);
    cout<<"original list:\n";
    for(int i=0;i<x.size();i++) cout<<x[i]<<" ";
    cout<<"\n";
    vector<int> sorted_list=simple_merge_sort(x);
    cout<<"sorted list:\n";
    for(int i=0;i<sorted_list.size();i++) cout<<sorted_list[i]<<" ";
    cout<<"\n";
    return 1;
}

