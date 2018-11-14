# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:00:33 2018

@author: Mingming
"""


## CUSTOMIZED MODULE TO FORMAT DATA




#%%
"""
Given a list, this function is removing the empty elements in the list and return a list with all the 
real content in there.
Usage:
    (wanted_list, wanted_index) = removeEnptyList(inputlist, unwanted_list)
    unwanted_list, is an optional input, default option is to remove empty list

INPUT:
    inputlist:     the list one wants to operate 
    unwanted_list: the list that we don't want
OUTPUT:
    wanted_list:   the returned list with unwanted contents removed
    wanted_index   the index from the original list, where there is wanted contents
"""
def removeList(*args):
    
    for count, content in enumerate(args):
        if count == 0:
           inputlist     = content
           unwanted_list = ''  # default option is to remove empty list
        elif count == 1:
            unwanted_list = content
    
    #unwanted_list = '\n' 
    wanted_index  = []  # define an empty list to store the cleaned HEX data
    wanted_list   = []
    for num in range(0, len(inputlist)):
        # get the content when it is not empty
        if inputlist[num] != unwanted_list:       
            wanted_index.append(num) 
            wanted_list.append(inputlist[num])
            
    return(wanted_list, wanted_index)



#%%
"""
Given a long string, return a new string with the unwanted characters deleted
Usage:
    new_str = deleteStr(input_str, unwanted_cha)
    
INPUT:
    input_str: the input string
    unwanted_cha: unwanted characters, a string
OUTPUT:
    new_str: the output with the unwanted string deleted, 
             this is a string
"""
def deleteStr(input_str, unwanted_cha):
    
    input_str_sep = input_str.split(unwanted_cha)   # spilt the larger string with the given unwanted string
    new_str_list, _ = removeList( input_str_sep )   # remove empty list
    new_str = new_str_list[0] # there is only one content in the list, just get the string and return it, instead of a list
    return new_str    
    

#%%
"""
Convert heximal number to signed integer number
Usage:
deci_data = Hex2DecSign (Hex, bits)

INPUT:
    Hex: the input hex number that needs to be computed,
         needs to be string
    bits: integers, indicate the bits of the input signal

OUTPUT:
    sign_deci: the output signed integer data 
               only return a signed decimal number if the input bits is correct;
               will return an error message if the input bits is wrong.
"""
def Hex2DecSign (Hex, bits):    
    # create a python dictionary to put all the bits converter here
    switcher = {
    8: bits_8(Hex, bits),
    16: bits_16(Hex, bits),
    24: bits_24(Hex, bits),
    32: bits_32(Hex, bits),
    64: bits_64(Hex, bits),
    }
    # using the dictionary .get() method to
    # return the signed integer, or the "invalid input bits" message
    sign_deci = switcher.get(bits,
                             "Invalid input bits, only take 8, 16, 24, 32, 64")
    # return the value if the input bits is correct, and calculate was done
    # print the error message if the bits is wrong, no calculation was done
    if isinstance(sign_deci, int): 
        return sign_deci
    else:
        print(sign_deci)      

def bits_8(Hex, bits):   
    deci_signed = int(Hex , 16) - int('ff' , 16) 
    return deci_signed

def bits_16(Hex, bits):   
    deci_signed = int(Hex , 16) - int('ffff' , 16) 
    return deci_signed
    
def bits_24(Hex, bits):   
    deci_signed = int(Hex , 16) - int('ffffff' , 16) 
    return deci_signed  

def bits_32(Hex, bits):
    # convert the hex number into  signed decimal number
    # operate Two's complement to convert the original HEX number
    # into signed decimal number    
    deci_signed = int(Hex , 16) - int('ffffffff' , 16)     
    # 'ffffff' there are three bytes of data, each byte is 8 bits
    # Thus, each 'f' occupy one byte, 4 bits,
    # and then the number here is 24 bits
    return deci_signed
    
def bits_64(Hex, bits):   
    deci_signed = int(Hex , 16) - int('ffffffffffffffff' , 16) 
    return deci_signed



#%%
"""
This series of function turns the input of HEX number into a signed decimal number
Usage:
    my_signed_number = hex2data(raw_bytes)

INPUT:
    raw_bytes: bytes of HEX number, as a string input, i.e.'FFF5B0'
OUTPUT:
    my_signed_number: the signed decimal number, an int number
"""
def hex2dec(hex_bytes):   
    binary_bits = bin(int(hex_bytes,16))   # convert raw HEX into binary bits        
    # sometimes, the first two bits are '0b' to indicate it is a binary format number
    if binary_bits[0:2] == '0b':
        real_bytes = binary_bits[2:]
    else:
        real_bytes = binary_bits
    
    # the most significant byte might be less than 4 bits, then the string will only have
    # one or two bits for displaying (up to the bit with value of '1')
    # Thus, the first bit here is not the sign bit. 
    if (len(real_bytes)%4) == 0:
        sign_bit = real_bytes[0] # the first bit is the first bit of actural number                                
    else:
        sign_bit = '0'
    
    # this is a positive number
    if sign_bit == '0':
       my_signed_number = int(real_bytes, 2) 
    # this is a negative number
    elif sign_bit == '1':
         # conduct two's complement
         output_bits = reverse_bits(real_bytes)  
         my_signed_number = - ( int(output_bits, 2) + 1 )
     
    return my_signed_number   

  
"""
For a binary input data, reverse each bit. 
Usage:
    output_bits = reverse_bits(input_bits)
    
INPUT:
    input_bits: the input binary data, a string
OUTPUT:
    output_bits: the output binary data with each bit reversed,
    a string.        
"""        
def reverse_bits(input_bits):   
    output_bits = ""
    for bit in input_bits:        
        output_bits += reverse_1_bit(bit)
    return output_bits  

"""
Given one bit of binary data, reverse this bit. 
"""
def reverse_1_bit(input_bit):      
    if input_bit == '1':
        output_bit = '0'
    elif input_bit == '0':
        output_bit = '1'
    return output_bit

#%%    



