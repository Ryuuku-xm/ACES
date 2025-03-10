def get_result_from_raw(gid, uid, data):
    """
    find result by gid and uid
    :param gid: group id
    :param uid: user id
    :return: user point
    """
    group_data = data.get(gid)
    if group_data:
        user_point = group_data.get(uid)
        if user_point is not None:
            return user_point.get("result")
        else:
            print("Can't find uid {} in the group {}".format(uid, gid))
            return -1
    else:
        print("Can't find gid {}".format(gid))
        return -1

def get_data_from_raw(gid, uid, data):
    """
    find result by gid and uid
    :param gid: group id
    :param uid: user id
    :return: user data
    """
    group_data = data.get(gid)
    if group_data:
        user_point = group_data.get(uid)
        if user_point is not None:
            return user_point.get("data")
        else:
            print("Can't find uid {} in the group {}".format(uid, gid))
            return -1
    else:
        print("Can't find gid {}".format(gid))
        return -1

def get_time_from_raw(gid, uid, data):
    """
    find result by gid and uid
    :param gid: group id
    :param uid: user id
    :return: user time
    """
    group_data = data.get(gid)
    if group_data:
        user_point = group_data.get(uid)
        if user_point is not None:
            return user_point.get("time")
        else:
            print("Can't find uid {} in the group {}".format(uid, gid))
            return -1
    else:
        print("Can't find gid {}".format(gid))
        return -1