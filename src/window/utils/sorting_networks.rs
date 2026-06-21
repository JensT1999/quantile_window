pub fn sorting_network_16<T>(data: &mut [T]) where
T: Ord +
Copy
{
    sorting_network_cas(data, 0, 15);
    sorting_network_cas(data, 1, 14);
    sorting_network_cas(data, 2, 13);
    sorting_network_cas(data, 3, 12);
    sorting_network_cas(data, 4, 11);
    sorting_network_cas(data, 5, 10);
    sorting_network_cas(data, 6, 9);
    sorting_network_cas(data, 7, 8);

    sorting_network_cas(data, 0, 5);
    sorting_network_cas(data, 1, 7);
    sorting_network_cas(data, 2, 6);
    sorting_network_cas(data, 3, 4);
    sorting_network_cas(data, 8, 14);
    sorting_network_cas(data, 9, 13);
    sorting_network_cas(data, 10, 15);
    sorting_network_cas(data, 11, 12);

    sorting_network_cas(data, 0, 2);
    sorting_network_cas(data, 1, 3);
    sorting_network_cas(data, 4, 8);
    sorting_network_cas(data, 5, 9);
    sorting_network_cas(data, 6, 10);
    sorting_network_cas(data, 7, 11);
    sorting_network_cas(data, 12, 14);
    sorting_network_cas(data, 13, 15);

    sorting_network_cas(data, 0, 1);
    sorting_network_cas(data, 2, 7);
    sorting_network_cas(data, 3, 5);
    sorting_network_cas(data, 4, 6);
    sorting_network_cas(data, 8, 13);
    sorting_network_cas(data, 9, 11);
    sorting_network_cas(data, 10, 12);
    sorting_network_cas(data, 14, 15);

    sorting_network_cas(data, 1, 3);
    sorting_network_cas(data, 2, 4);
    sorting_network_cas(data, 5, 10);
    sorting_network_cas(data, 6, 9);
    sorting_network_cas(data, 7, 8);
    sorting_network_cas(data, 11, 13);
    sorting_network_cas(data, 12, 14);

    sorting_network_cas(data, 1, 2);
    sorting_network_cas(data, 3, 4);
    sorting_network_cas(data, 5, 7);
    sorting_network_cas(data, 8, 10);
    sorting_network_cas(data, 11, 12);
    sorting_network_cas(data, 13, 14);

    sorting_network_cas(data, 2, 3);
    sorting_network_cas(data, 4, 6);
    sorting_network_cas(data, 9, 11);
    sorting_network_cas(data, 12, 13);

    sorting_network_cas(data, 4, 5);
    sorting_network_cas(data, 6, 7);
    sorting_network_cas(data, 8, 9);
    sorting_network_cas(data, 10, 11);

    sorting_network_cas(data, 3, 4);
    sorting_network_cas(data, 5, 6);
    sorting_network_cas(data, 7,8);
    sorting_network_cas(data, 9, 10);
    sorting_network_cas(data, 11, 12);

    sorting_network_cas(data, 6, 7);
    sorting_network_cas(data, 8, 9);
}

#[inline(always)]
fn sorting_network_cas<T>(data: &mut [T], index1: usize, index2: usize) where
T: Ord +
Copy
{
    let data_tup = unsafe {
        (*data.get_unchecked(index1), *data.get_unchecked(index2))
    };

    if data_tup.0 > data_tup.1 {
        unsafe {
            *data.get_unchecked_mut(index1) = data_tup.1;
            *data.get_unchecked_mut(index2) = data_tup.0;
        }
    }
}
