from models.loss.loss import get_sobel


def zero_flow_l1_loss(data, config=None, **kwargs):
    assert isinstance(data, list)
    assert len(data) == 2
    flow = data[0]
    gt = data[1]
    loss_map = (flow - gt) ** 2
    loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
    return loss_map

def zero_flow_l2_loss(data, config=None, **kwargs):
    assert isinstance(data, list)
    assert len(data) == 2
    flow = data[0]
    gt = data[1]
    loss_map = (flow - gt) ** 2
    return loss_map

def flow2_loss(data, config=None, **kwargs):
    assert isinstance(data, list)
    assert len(data) == 2
    return get_sobel()(data[0], data[1])
